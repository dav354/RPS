import os
import time
import numpy as np
from flask import Flask, Response, render_template, jsonify
from collections import deque
import cv2
import mediapipe as mp
import threading



from utils.game_logic import prepare_round, play_round, reset_game, game_manager
from utils.draw import draw_landmarks
from utils.gesture_buffer import GestureCollector
from utils.camera import setup_camera, get_display_frame
from utils.stats import get_system_stats
from utils.tpu import load_tpu_model, run_inference

# === Configuration ===
PEPPER_IP = os.environ.get("PEPPER_IP")
camera_source = f"http://{PEPPER_IP}:5001/video_feed" if PEPPER_IP else 0
label_map = ["none", "rock", "paper", "scissors"]
gesture_collector = GestureCollector(duration=2.0)
PREDICTION_HISTORY = deque(maxlen=5)

# === Load TPU ===
interpreter, TPU_OK = load_tpu_model()
input_details = interpreter.get_input_details() if interpreter else None
output_details = interpreter.get_output_details() if interpreter else None

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
    """
    if landmarks.shape[1] != 3:
        raise ValueError("Landmarks must be 3D (x, y, z) for 3D alignment.")

    # Translate the wrist to the origin
    translated_landmarks = landmarks - landmarks[0]

    # Calculate the vector from wrist (0) to middle finger base (9)
    v_wrist_mid_finger = translated_landmarks[9] - translated_landmarks[0]
    v_wrist_mid_finger_norm = np.linalg.norm(v_wrist_mid_finger)

    if v_wrist_mid_finger_norm < 1e-6:
        return translated_landmarks

    # Normalize this vector
    v_norm = v_wrist_mid_finger / v_wrist_mid_finger_norm

    # Target vector (aligning with Y-axis)
    target_y_axis = np.array([0, 1, 0])
    rotation_axis = np.cross(v_norm, target_y_axis)
    rotation_axis_norm = np.linalg.norm(rotation_axis)

    if rotation_axis_norm < 1e-6:
        if np.dot(v_norm, target_y_axis) < 0:
            rotation_angle = np.pi
            rotation_axis = np.array([1, 0, 0])
        else:
            return translated_landmarks
    else:
        rotation_axis = rotation_axis / rotation_axis_norm
        rotation_angle = np.arccos(np.dot(v_norm, target_y_axis))

    # Rodrigues' rotation formula
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
    landmark_pts_2d = None
    results = mp_hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Extract 3D coordinates
        coords_3d = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
        coords_3d_aligned = align_landmarks_3d(coords_3d)

        # Normalize coordinates
        coords_min = coords_3d_aligned.min(axis=0)
        coords_max = coords_3d_aligned.max(axis=0)
        coords_range = coords_max - coords_min
        coords_normalized = (coords_3d_aligned - coords_min) / np.where(coords_range > 0, coords_range, 1)
        
        flat_coords = coords_normalized.flatten()
        
        # Get 2D pixel coordinates for drawing
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
                if gesture_collector.collecting:
                    gesture_collector.add_gesture(current_gesture)
            else:
                gesture = "Unknown"
                confidence = conf
        else:
            gesture = "Inference Error"

    return gesture, confidence, infer_ms, landmark_pts_2d


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
        landmark_pts_to_draw = None

        if not TPU_OK:
            gesture = "TPU Offline"
        elif gesture_collector.collecting:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gesture, confidence, infer_ms, landmark_pts_to_draw = _process_hand_gestures(rgb_frame)
            if landmark_pts_to_draw is not None:
                draw_landmarks(frame, landmark_pts_to_draw, frame.shape[1], frame.shape[0])
        elif game_manager.state['game_over']:
             gesture = "Game Over"
        elif game_manager.round > 0:
             gesture = "Next round..."
        else:
            gesture = "Idle"
            
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

def run_game_flow():
    """
    Manages the entire game flow of 3 rounds sequentially.
    This function runs in a background thread.
    """
    log("[🚀] Starting new game flow...")

    for i in range(game_manager.total_rounds):
        if game_manager.state["game_over"]:
            log(f"[🏁] Game ended early on round {i} due to score.")
            break

        # 1. Prepare the round
        prepare_round()
        log(f"[🎬] Round {game_manager.state['round']}/{game_manager.total_rounds} started.")

        # 2. Collect gestures for a fixed duration
        gesture_collector.start()
        log(f"[👂] Collecting gestures for {gesture_collector.duration} seconds...")
        time.sleep(gesture_collector.duration)

        # 3. Finalize the round with the most common gesture
        final_gesture = gesture_collector.get_most_common()
        log(f"[🧠] Round decided. Most common gesture: {final_gesture}")
        play_round(final_gesture)
        gesture_collector.reset() # Stop collecting and clear buffer

        # 4. Check if the game is over after the round
        if game_manager.state["game_over"]:
            log("[🏁] Game over condition met.")
            break
        
        # 5. Wait before the next round for better user experience
        log("[⏳] Cooldown before next round...")
        time.sleep(3)

    # Final check to ensure game over state is set if all rounds are played
    if not game_manager.state["game_over"]:
        game_manager._check_game_over(force_end=True)

    log("[🎉] Game flow finished.")


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


@app.route("/start_game")
def start_game_route_api():
    if not TPU_OK:
        log("[❌] Cannot start game: TPU not available.")
        return jsonify({"status": "no_tpu", "message": "TPU not available."})

    if not cap.isOpened():
        log("[❌] Cannot start game: Camera not available.")
        return jsonify({"status": "no_camera", "message": "Camera not available."})

    # Check if a game is already in progress
    if gesture_collector.collecting or (game_manager.round > 0 and not game_manager.state["game_over"]):
        log("[⚠️] Cannot start new game: A game is already in progress.")
        return jsonify({"status": "already_running", "message": "A game is already in progress."})

    # Reset game state for a fresh start
    reset_game()

    # Launch the main game flow in a background thread
    threading.Thread(target=run_game_flow, daemon=True).start()
    return jsonify({"status": "started", "message": "New game started in background."})


@app.route("/game_state")
def get_game_state_route():
    return jsonify(game_manager.state)


@app.route("/reset_game")
def reset_game_route_api():
    # The game logic thread will see the 'game_over' flag and stop itself.
    reset_game()
    gesture_collector.reset()
    log("[🔄] Game has been reset.")
    return jsonify({"status": "reset", "message": "Game reset."})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, threaded=True, debug=False)