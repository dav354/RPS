import os
import time
import numpy as np
from flask import Flask, Response, render_template, jsonify
from collections import deque
import cv2
import mediapipe as mp
import logging

from utils.game_logic import prepare_round, game_state, play_round, reset_game
from utils.draw import draw_landmarks
from utils.gesture_buffer import GestureCollector
from utils.camera import setup_camera, get_display_frame
from utils.stats import get_system_stats
from utils.tpu import load_tpu_model, run_inference

# === Configuration ===
PEPPER_IP = os.environ.get("PEPPER_IP")
camera_source = f"{PEPPER_IP}/video_feed"
label_map = ["none", "rock", "paper", "scissors"]
gesture_collector = GestureCollector(duration=2.0)
PREDICTION_HISTORY = deque(maxlen=5)

# === Load TPU ===
interpreter, TPU_OK = load_tpu_model()
input_details = interpreter.get_input_details() if interpreter else None
output_details = interpreter.get_output_details() if interpreter else None

# === Setup Logging ===
logging.getLogger('werkzeug').setLevel(logging.WARNING)

# === Setup camera ===
cap = setup_camera(camera_source)

# === Setup MediaPipe ===
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# === System Stats ===
latest_stats = {
    "gesture": "Initializing...",
    "confidence": 0.0,
    "fps": 0.0,
    "cpu": 0.0,
    "ram": "0MB / 0MB",
    "inference_ms": 0.0,
    "cpu_temp": 0.0,
    "tpu": TPU_OK,
}

# === Flask App ===
app = Flask(__name__)


def log(msg: str):
    print(msg)

def align_landmarks_3d(landmarks):
    """
    Aligns 3D hand landmarks:
    1. Translates the wrist (landmark 0) to the origin.
    2. Rotates the hand to a more canonical orientation.
       This example uses the vector from wrist (0) to middle finger base (9) for alignment.
       A more robust alignment might involve more points or PCA.
    """
    if landmarks.shape[1] != 3:
        raise ValueError("Landmarks must be 3D (x, y, z) for 3D alignment.")

    # Translate the wrist to the origin (landmark 0 is the wrist)
    translated_landmarks = landmarks - landmarks[0]

    # Calculate the vector from wrist (0) to middle finger base (9)
    # This vector will be aligned with the Y-axis (or similar)
    v_wrist_mid_finger = translated_landmarks[9] - translated_landmarks[0]
    v_wrist_mid_finger_norm = np.linalg.norm(v_wrist_mid_finger)

    if v_wrist_mid_finger_norm < 1e-6: # Avoid division by zero
        return translated_landmarks

    # Normalize this vector
    v_norm = v_wrist_mid_finger / v_wrist_mid_finger_norm

    # Create target vectors (e.g., align v_norm with the Y-axis)
    target_y_axis = np.array([0, 1, 0])
    # Compute the rotation axis and angle
    rotation_axis = np.cross(v_norm, target_y_axis)
    rotation_axis_norm = np.linalg.norm(rotation_axis)

    if rotation_axis_norm < 1e-6: # Vectors are already aligned or opposite
        if np.dot(v_norm, target_y_axis) < 0: # Opposite, rotate 180 degrees
            rotation_angle = np.pi
            rotation_axis = np.array([1, 0, 0]) # Arbitrary axis for 180 deg
        else:
            return translated_landmarks # Already aligned
    else:
        rotation_axis = rotation_axis / rotation_axis_norm
        rotation_angle = np.arccos(np.dot(v_norm, target_y_axis))

    # Build rotation matrix using Rodrigues' rotation formula
    K = np.array([
        [0, -rotation_axis[2], rotation_axis[1]],
        [rotation_axis[2], 0, -rotation_axis[0]],
        [-rotation_axis[1], rotation_axis[0], 0]
    ])
    R = np.eye(3) + np.sin(rotation_angle) * K + (1 - np.cos(rotation_angle)) * np.dot(K, K)

    # Apply rotation
    aligned_landmarks = np.dot(translated_landmarks, R.T)

    return aligned_landmarks

def _process_hand_gestures(rgb_frame):
    gesture, confidence, infer_ms = "No Hand", 0.0, 0.0
    landmark_pts_2d = None # Will be used for drawing
    results = mp_hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        # --- KEY CHANGE: Extracting x, y, and z coordinates ---
        coords_3d = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)

        # Apply 3D alignment, mirroring the training script
        coords_3d_aligned = align_landmarks_3d(coords_3d)

        # Normalize coordinates to [0, 1] range after alignment, mirroring training script
        coords_min = coords_3d_aligned.min(axis=0)
        coords_max = coords_3d_aligned.max(axis=0)
        coords_range = coords_max - coords_min

        # Avoid division by zero for dimensions with no variation
        coords_normalized = (coords_3d_aligned - coords_min) / np.where(coords_range > 0, coords_range, 1)

        # Flatten the 3D coordinates for the model input
        flat_coords = coords_normalized.flatten()

        # For drawing, you might want to use the original 2D (x,y) from the raw landmarks
        landmark_pts_2d = np.array([[lm.x * rgb_frame.shape[1], lm.y * rgb_frame.shape[0]] for lm in hand_landmarks.landmark], dtype=np.int32)


        dequantized_preds, current_infer_ms = run_inference(interpreter, input_details, output_details, flat_coords)
        infer_ms = current_infer_ms

        if dequantized_preds is not None:
            PREDICTION_HISTORY.append(dequantized_preds)
            avg_pred = np.mean(PREDICTION_HISTORY, axis=0)
            idx = int(np.argmax(avg_pred))
            conf = float(avg_pred[idx])

            if conf > 0.75:
                gesture = label_map[idx]
            else:
                gesture = "Unknown"
                confidence = conf
        else:
            gesture = "Inference Error"
            confidence = 0.0
    else:
        gesture = "No Hand"
        confidence = 0.0

    if gesture_collector.collecting:
        gesture_collector.add_gesture(gesture)

    return gesture, confidence, infer_ms, landmark_pts_2d



def _check_and_finalize_round():
    if gesture_collector.is_done():
        final_gesture = gesture_collector.get_most_common()
        log(f"[🧠] Round decided. Most common gesture: {final_gesture}")
        play_round(final_gesture)
        gesture_collector.reset()


def generate_frames():
    prev_time = time.time()

    while True:
        frame_ok, frame = get_display_frame(cap, camera_source)

        if not frame_ok:
            ok_enc, buf = cv2.imencode(".jpg", frame)
            if ok_enc:
                yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
                )
            time.sleep(1)
            continue

        current_time = time.time()
        delta_time = current_time - prev_time
        fps = 1.0 / delta_time if delta_time > 0 else 0
        prev_time = current_time

        cpu, ram_str, cpu_temp = get_system_stats()
        gesture = "N/A"
        confidence = 0.0
        infer_ms = 0.0
        landmark_pts_to_draw = None # Renamed to avoid confusion with internal 3D coords

        if not TPU_OK:
            cv2.putText(frame, "TPU NOT AVAILABLE", (10, frame.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            gesture = "TPU Offline"
        elif not gesture_collector.collecting:
            gesture = "Idle"
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gesture, confidence, infer_ms, landmark_pts_to_draw = _process_hand_gestures(rgb_frame)
            if landmark_pts_to_draw is not None:
                # Pass the original frame dimensions for drawing
                draw_landmarks(frame, landmark_pts_to_draw, frame.shape[1], frame.shape[0])
            _check_and_finalize_round()

        latest_stats.update({
            "gesture": gesture,
            "confidence": round(confidence, 2),
            "fps": round(fps, 1),
            "cpu": cpu,
            "ram": ram_str,
            "inference_ms": round(infer_ms, 1),
            "cpu_temp": cpu_temp,
            "tpu": TPU_OK,
            "camera": cap.isOpened(),
        })

        ok_enc, buf = cv2.imencode(".jpg", frame)
        if not ok_enc:
            log("[⚠️] JPEG encoding failed.")
            continue

        yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        )


# === Routes ===
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed_route():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/gesture_data")
def gesture_data_route():
    return jsonify(latest_stats)


@app.route("/start_round")
def start_round_route_api():
    if not TPU_OK:
        log("[❌] Cannot start round: TPU not available.")
        return jsonify({"status": "no_tpu", "message": "TPU not available. Cannot start round."})

    if not cap.isOpened():
        log("[❌] Cannot start round: Camera not available.")
        return jsonify({"status": "no_camera", "message": "Camera not available. Cannot start round."})

    if not gesture_collector.collecting:
        gesture_collector.start()
        prepare_round()
        log("[🎬] Round started: collecting gestures")
        return jsonify({"status": "collecting", "message": "Round started. Collecting gestures..."})
    else:
        return jsonify({"status": "already_collecting", "message": "Already collecting gestures."})


@app.route("/game_state")
def get_game_state_route():
    return jsonify(game_state)


@app.route("/reset_game")
def reset_game_route_api():
    reset_game()
    gesture_collector.reset()
    log("[🔄] Game has been reset.")
    return jsonify({"status": "reset", "message": "Game reset."})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, threaded=True, debug=False)