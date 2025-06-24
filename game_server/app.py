#!/usr/bin/env python3
import cv2
import time
import os
import numpy as np
import psutil
import mediapipe as mp
import threading
from flask import Flask, Response, render_template, jsonify
from collections import deque

from draw import draw_landmarks
from game_logic import game_state, play_round, reset_game, prepare_round
from gesture_buffer import GestureCollector

# tflite_runtime imports for TPU
from tflite_runtime.interpreter import Interpreter, load_delegate

# === Configuration ===
PEPPER_IP = os.environ.get("PEPPER_IP")
label_map = ["none", "rock", "paper", "scissors"]
gesture_collector = GestureCollector(duration=2.0)
PREDICTION_HISTORY = deque(maxlen=5)

# === TPU Initialization ===
TPU_OK = False
interpreter = None
input_details = None
output_details = None

try:
    interpreter = Interpreter(
        model_path="models/model_edgetpu.tflite",
        experimental_delegates=[load_delegate("libedgetpu.so.1")],
    )
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    TPU_OK = True
    print("✅ TPU initialized successfully.")
except Exception as e:
    print(f"⚠️ TPU initialization failed: {e}")
    interpreter = None

latest_stats = {
    "gesture": "Initializing...",
    "confidence": 0.0,
    "fps": 0.0,
    "cpu": 0.0,
    "ram": "0MB / 0MB",
    "inference_ms": 0.0,
    "cpu_temp": 0.0,
    "tpu": TPU_OK,
    "camera": False,
}

# === MediaPipe Hand Detection Setup ===
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# === Flask App ===
app = Flask(__name__)

def log(msg: str):
    print(msg)

class CameraThread(threading.Thread):
    def __init__(self, src=0):
        super().__init__()
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.frame = None
        self.is_running = False
        self.lock = threading.Lock()

    def run(self):
        self.is_running = True
        while self.is_running:
            if not self.cap.isOpened():
                log("[⚠️] Camera not open. Attempting to reconnect in 2s...")
                time.sleep(2)
                self.cap.release()
                self.cap.open(self.src)
                continue

            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = cv2.flip(frame, 1)
            else:
                log("[⚠️] Frame capture failed. Will retry.")
                time.sleep(0.5) # Don't spam retries on failure

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def is_opened(self):
        return self.cap.isOpened()

    def stop(self):
        self.is_running = False
        self.cap.release()

# === Initialize and Start Camera Thread ===
camera_source = f"{PEPPER_IP}/video_feed"
log(f"📹 Starting camera thread for: {camera_source}")
camera_thread = CameraThread(src=camera_source)
camera_thread.daemon = True
camera_thread.start()

def get_system_stats():
    cpu = psutil.cpu_percent(interval=None)
    vm = psutil.virtual_memory()
    ram_str = f"{vm.used // (1024*1024)}MB / {vm.total // (1024*1024)}MB"
    cpu_temp = 0.0
    try:
        temps = psutil.sensors_temperatures()
        for label in ("cpu_thermal", "cpu-thermal", "coretemp"):
            if label in temps and temps[label]:
                cpu_temp = temps[label][0].current
                break
    except Exception:
        pass # Not all systems have temp sensors available
    return cpu, ram_str, round(cpu_temp, 1)

def run_inference(coords_normalized_scaled):
    # This function is computationally intensive but not blocking I/O, so it's okay here
    if not TPU_OK or interpreter is None:
        return None, 0.0

    inp = coords_normalized_scaled.flatten().astype(np.float32)
    t0 = time.time()

    if 'quantization' in input_details[0] and input_details[0]['quantization'][0] != 0:
        in_scale, in_zero_point = input_details[0]["quantization"]
        quantized_input = (inp / in_scale + in_zero_point).astype(input_details[0]["dtype"])
    else:
        quantized_input = inp.astype(input_details[0]["dtype"])

    interpreter.set_tensor(input_details[0]["index"], [quantized_input])
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])[0]
    infer_ms = (time.time() - t0) * 1000

    if 'quantization' in output_details[0] and output_details[0]['quantization'][0] != 0:
        out_scale, out_zero_point = output_details[0]["quantization"]
        dequantized_output = (output_data.astype(np.float32) - out_zero_point) * out_scale
    else:
        dequantized_output = output_data.astype(np.float32)
        
    return dequantized_output, infer_ms

# === Helper functions for gesture processing ===
def _process_hand_gestures(rgb_frame):
    gesture, confidence, infer_ms = "No Hand", 0.0, 0.0
    landmark_pts = None
    results = mp_hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        landmark_pts = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark], dtype=np.float32)
        
        coords_normalized = landmark_pts - landmark_pts.mean(axis=0)
        max_abs_val = np.max(np.abs(coords_normalized))
        coords_scaled = coords_normalized / (max_abs_val if max_abs_val > 0 else 1)

        dequantized_preds, current_infer_ms = run_inference(coords_scaled)
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
            
    return gesture, confidence, infer_ms, landmark_pts

def _check_and_finalize_round():
    if gesture_collector.is_done():
        final_gesture = gesture_collector.get_most_common()
        log(f"[🧠] Round decided. Most common gesture: {final_gesture}")
        play_round(final_gesture)
        gesture_collector.reset()

def generate_frames():
    prev_time = time.time()
    round_prepared = False

    while True:
        frame = camera_thread.read()
        
        if frame is None:
            # Create an error frame if camera thread hasn't captured anything yet
            error_frame_img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_frame_img, "CONNECTING TO CAMERA...", (50, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ok_enc, buf = cv2.imencode(".jpg", error_frame_img)
            if ok_enc:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
            time.sleep(0.5)
            continue
        
        # --- Frame timing and system stats ---
        current_time = time.time()
        delta_time = current_time - prev_time
        fps = 1.0 / delta_time if delta_time > 0 else 0
        prev_time = current_time
        cpu, ram_str, cpu_temp = get_system_stats()

        # --- Gesture processing logic ---
        gesture, confidence, infer_ms, landmark_pts = "Idle", 0.0, 0.0, None

        if not TPU_OK:
            gesture = "TPU Offline"
        elif gesture_collector.collecting:
            # Prepare round once
            if not round_prepared:
                prepare_round()
                round_prepared = True
                
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gesture, confidence, infer_ms, landmark_pts = _process_hand_gestures(rgb_frame)
            _check_and_finalize_round()
        else:
            round_prepared = False # Reset for the next round

        # --- Draw landmarks if found ---
        if landmark_pts is not None:
            draw_landmarks(frame, landmark_pts, frame.shape[1], frame.shape[0])

        # --- Update latest stats for the web UI ---
        latest_stats.update({
            "gesture": gesture,
            "confidence": round(confidence, 2),
            "fps": round(fps, 1),
            "cpu": cpu, "ram": ram_str,
            "inference_ms": round(infer_ms, 1),
            "cpu_temp": cpu_temp, "tpu": TPU_OK,
            "camera": camera_thread.is_opened(),
        })

        # --- Encode and yield the frame ---
        ok_enc, buf = cv2.imencode(".jpg", frame)
        if ok_enc:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

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
        return jsonify({"status": "no_tpu", "message": "TPU not available."})
    if not camera_thread.is_opened():
        log("[❌] Cannot start round: Camera not available.")
        return jsonify({"status": "no_camera", "message": "Camera not available."})
    if not gesture_collector.collecting:
        gesture_collector.start()
        log("[🎬] Round started: collecting gestures")
        return jsonify({"status": "collecting", "message": "Round started."})
    else:
        return jsonify({"status": "already_collecting", "message": "Already collecting."})

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
    try:
        app.run(host="0.0.0.0", port=80, threaded=True, debug=False)
    finally:
        log("Stopping camera thread...")
        camera_thread.stop()
        camera_thread.join() # Wait for thread to finish