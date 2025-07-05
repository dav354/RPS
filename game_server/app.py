import os
import time
import numpy as np
from flask import Flask, Response, render_template, jsonify, request
from collections import deque
import cv2
import mediapipe as mp
import logging

# --- NEW: Import the new, robust game logic functions ---
from utils.game_logic import (
    add_gesture_from_camera,
    get_game_state_for_frontend,
    start_new_round_from_frontend,
    reset_game_from_frontend
)
from utils.draw import draw_landmarks
from utils.camera import setup_camera, get_display_frame
from utils.stats import get_system_stats
from utils.tpu import load_tpu_model, run_inference

# === Configuration ===
PEPPER_IP = os.environ.get("PEPPER_IP")
camera_source = f"{PEPPER_IP}/video_feed"
label_map = ["none", "rock", "paper", "scissors"]
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
    """Aligns 3D hand landmarks."""
    if landmarks.shape[1] != 3:
        raise ValueError("Landmarks must be 3D (x, y, z) for 3D alignment.")

    translated_landmarks = landmarks - landmarks[0]
    v_wrist_mid_finger = translated_landmarks[9] - translated_landmarks[0]
    v_wrist_mid_finger_norm = np.linalg.norm(v_wrist_mid_finger)

    if v_wrist_mid_finger_norm < 1e-6:
        return translated_landmarks

    v_norm = v_wrist_mid_finger / v_wrist_mid_finger_norm
    target_y_axis = np.array([0, 1, 0])
    rotation_axis = np.cross(v_norm, target_y_axis)
    rotation_axis_norm = np.linalg.norm(rotation_axis)

    if rotation_axis_norm < 1e-6:
        return translated_landmarks

    rotation_axis = rotation_axis / rotation_axis_norm
    rotation_angle = np.arccos(np.dot(v_norm, target_y_axis))
    K = np.array([[0, -rotation_axis[2], rotation_axis[1]], [rotation_axis[2], 0, -rotation_axis[0]], [-rotation_axis[1], rotation_axis[0], 0]])
    R = np.eye(3) + np.sin(rotation_angle) * K + (1 - np.cos(rotation_angle)) * np.dot(K, K)
    aligned_landmarks = np.dot(translated_landmarks, R.T)
    return aligned_landmarks

def _process_hand_gestures(rgb_frame):
    """Processes a frame to detect a gesture and sends it to the game manager."""
    gesture, confidence, infer_ms = "No Hand", 0.0, 0.0
    landmark_pts_2d = None
    results = mp_hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        coords_3d = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
        coords_3d_aligned = align_landmarks_3d(coords_3d)
        coords_min, coords_max = coords_3d_aligned.min(axis=0), coords_3d_aligned.max(axis=0)
        coords_range = coords_max - coords_min
        coords_normalized = (coords_3d_aligned - coords_min) / np.where(coords_range > 0, coords_range, 1)
        flat_coords = coords_normalized.flatten()
        landmark_pts_2d = np.array([[lm.x * rgb_frame.shape[1], lm.y * rgb_frame.shape[0]] for lm in hand_landmarks.landmark], dtype=np.int32)

        dequantized_preds, current_infer_ms = run_inference(interpreter, input_details, output_details, flat_coords)
        infer_ms = current_infer_ms

        if dequantized_preds is not None:
            PREDICTION_HISTORY.append(dequantized_preds)
            avg_pred = np.mean(PREDICTION_HISTORY, axis=0)
            idx = int(np.argmax(avg_pred))
            conf = float(avg_pred[idx])

            if conf > 0.75:
                current_gesture = label_map[idx]
                gesture = current_gesture
                confidence = conf
                # --- NEW: Forward the detected gesture to the game manager ---
                add_gesture_from_camera(current_gesture)
            else:
                gesture = "Unknown"
                confidence = conf
                add_gesture_from_camera(gesture) # Also forward "Unknown"
        else:
            gesture = "Inference Error"

    return gesture, confidence, infer_ms, landmark_pts_2d


def generate_frames():
    """Main loop for capturing camera frames and processing gestures."""
    prev_time = time.time()

    while True:
        frame_ok, frame = get_display_frame(cap, camera_source)

        if not frame_ok:
            ok_enc, buf = cv2.imencode(".jpg", frame)
            if ok_enc:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
            time.sleep(1)
            continue

        current_time = time.time()
        delta_time = current_time - prev_time
        fps = 1.0 / delta_time if delta_time > 0 else 0
        prev_time = current_time

        cpu, ram_str, cpu_temp = get_system_stats()
        gesture, confidence, infer_ms = "N/A", 0.0, 0.0
        landmark_pts_to_draw = None

        if not TPU_OK:
            gesture = "TPU Offline"
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gesture, confidence, infer_ms, landmark_pts_to_draw = _process_hand_gestures(rgb_frame)
            if landmark_pts_to_draw is not None:
                draw_landmarks(frame, landmark_pts_to_draw, frame.shape[1], frame.shape[0])

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
            continue

        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")


# === NEW/UPDATED Routes ===
@app.route("/")
def index():
    """Serves the main HTML page."""
    return render_template("index.html")


@app.route("/video_feed")
def video_feed_route():
    """Provides the camera video stream."""
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/gesture_data")
def gesture_data_route():
    """Provides raw stats for the UI, but is not used for game logic."""
    return jsonify(latest_stats)


@app.route("/get_game_state")
def get_game_state_route():
    """Provides the authoritative game state from the game manager."""
    return jsonify(get_game_state_for_frontend())


@app.route("/add_gesture", methods=['POST'])
def add_gesture_route():
    """Receives gestures from the frontend and passes them to the game manager."""
    data = request.get_json()
    gesture = data.get('gesture')
    if gesture:
        add_gesture_from_camera(gesture)
        return jsonify({"status": "received"}), 200
    return jsonify({"status": "error", "message": "No gesture provided"}), 400


@app.route("/start_new_round")
def start_new_round_route():
    """Tells the game manager to start a new round."""
    start_new_round_from_frontend()
    return jsonify({"status": "starting_round"})


@app.route("/reset_game")
def reset_game_route():
    """Tells the game manager to reset the game."""
    reset_game_from_frontend()
    return jsonify({"status": "reset"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, threaded=True, debug=False)
