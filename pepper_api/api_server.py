#!/usr/bin/env python2

import qi
import time
import cv2
import numpy as np
from flask import Flask, Response, request, jsonify
import threading

app = Flask(__name__)

# --- NAOqi Service Connection ---
try:
    session = qi.Session()
    session.connect("tcp://127.0.0.1:9559")

    video_service = session.service("ALVideoDevice")
    motion_service = session.service("ALMotion")
    posture_service = session.service("ALRobotPosture")
    tts_service = session.service("ALTextToSpeech")
    asr_service = session.service("ALSpeechRecognition")
    memory_service = session.service("ALMemory")
    life_service = session.service("ALAutonomousLife")
    awareness_service = session.service("BasicAwareness")
    print("[INFO] Successfully connected to NAOqi services.")

except Exception as e:
    print("[ERROR] Error connecting to NAOqi services:", e)
    exit(1)

GESTURE_HOLD_DURATION = 4

# --- Integrated Head Control Logic ---
def freeze_head():
    """Sets head stiffness to 1.0 to lock it in place. This is a blocking call."""
    print("[ACTION] Freezing head position.")
    try:
        motion_service.setStiffnesses(["HeadYaw", "HeadPitch"], 1.0)
    except Exception as e:
        print("[ERROR] Could not freeze head:", e)

def unfreeze_head():
    """Sets head stiffness to 0.0 to allow it to move freely. This is a blocking call."""
    print("[ACTION] Unfreezing head position.")
    try:
        motion_service.setStiffnesses(["HeadYaw", "HeadPitch"], 0.0)
    except Exception as e:
        print("[ERROR] Could not unfreeze head:", e)


# --- Autonomous Behavior Control ---
def disable_autonomous_behaviors_and_freeze_head():
    """Disables built-in behaviors and freezes the head for stable operation."""
    try:
        if awareness_service.isAwarenessRunning():
            awareness_service.stopAwareness()
            print("[CONFIG] Basic Awareness disabled.")

        if life_service.getState() != "disabled":
            life_service.setState("disabled")
            print("[CONFIG] Autonomous Life disabled.")

        motion_service.setStiffnesses("Body", 1.0)
        print("[CONFIG] Body stiffness set to 1.0.")

        # Freeze the head on startup
        freeze_head()

    except Exception as e:
        print("[WARN] Error during autonomous behavior disabling:", e)

# --- Camera Setup ---
camera_name = "flask_cam"
name_id = None
def setup_camera():
    """Subscribes to the robot's top camera."""
    global name_id
    camera_index = 0  # top camera
    resolution = 1    # 640x480
    color_space = 13  # BGR
    fps = 20
    try:
        name_id = video_service.subscribeCamera(camera_name, camera_index, resolution, color_space, fps)
        print("[INFO] Camera subscribed successfully.")
    except Exception as e:
        print("[ERROR] Error subscribing to camera:", e)

def generate_video_stream():
    """Generates a multipart JPEG stream from the camera feed."""
    if name_id is None:
        return
    while True:
        try:
            image = video_service.getImageRemote(name_id)
            if image is None:
                time.sleep(1.0/fps)
                continue

            width, height, _, array = image[0], image[1], image[2], image[6]
            img = np.frombuffer(array, dtype=np.uint8).reshape((height, width, 3))
            ret, jpeg = cv2.imencode('.jpg', img)
            if not ret:
                continue

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
        except Exception as e:
            print("[WARN] Error getting image from camera:", e)
            time.sleep(1)

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return "Pepper API Server is running. Video at /video_feed"

# --- Threading Decorator ---
def run_in_thread(target_func):
    """Decorator to run a function in a separate, non-blocking thread."""
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=target_func, args=args, kwargs=kwargs)
        thread.daemon = True
        thread.start()
    return wrapper

# --- Robot Actions (Non-Blocking) ---

@run_in_thread
def do_rock():
    print("[ACTION] Performing: rock")
    motion_service.setStiffnesses("RArm", 1.0)
    motion_service.setAngles(["RShoulderPitch", "RElbowRoll", "RWristYaw", "RHand"],
                             [1.0, 0.5, 0.0, 0.0], 0.2)
    time.sleep(GESTURE_HOLD_DURATION)
    posture_service.goToPosture("StandInit", 0.5)
    # Head remains frozen

@run_in_thread
def do_paper():
    print("[ACTION] Performing: paper")
    motion_service.setStiffnesses("RArm", 1.0)
    motion_service.setAngles(["RShoulderPitch", "RShoulderRoll", "RElbowRoll", "RWristYaw", "RHand"],
                             [0.8, 0.0, 1.0, -1.2, 1.0], 0.2)
    time.sleep(GESTURE_HOLD_DURATION)
    posture_service.goToPosture("StandInit", 0.5)
    # Head remains frozen

@run_in_thread
def do_scissors():
    print("[ACTION] Performing: scissors")
    motion_service.setStiffnesses("RArm", 1.0)
    motion_service.setAngles(["RShoulderPitch", "RShoulderRoll", "RElbowRoll", "RWristYaw", "RHand"],
                             [0.8, 0.0, 1.0, 0.0, 1.0], 0.2)
    time.sleep(GESTURE_HOLD_DURATION)
    posture_service.goToPosture("StandInit", 0.5)
    # Head remains frozen

@run_in_thread
def do_swing():
    print("[ACTION] Performing: swing")
    # Head is already frozen from startup
    posture_service.goToPosture("StandInit", 0.5)
    motion_service.setStiffnesses("RArm", 1.0)
    # A short loop for the swing animation
    for _ in range(2):
        motion_service.setAngles(["RShoulderPitch", "RElbowRoll"], [0.4, 1.1], 0.3)
        time.sleep(0.3)
        motion_service.setAngles(["RShoulderPitch", "RElbowRoll"], [1.3, 0.4], 0.3)
        time.sleep(0.3)

@run_in_thread
def say_text_threaded(text):
    try:
        print("[ACTION] Saying:", text)
        tts_service.say(str(text))
    except Exception as e:
        print("[ERROR] TTS Error:", e)

# --- Flask Routes for Actions ---

@app.route('/gesture/<gesture_name>')
def perform_gesture(gesture_name):
    gesture_map = {
        "rock": do_rock,
        "paper": do_paper,
        "scissors": do_scissors,
        "swing": do_swing
    }
    func = gesture_map.get(gesture_name.lower())
    if func:
        func()
        return jsonify({"status": "ok", "message": "Gesture " + gesture_name + " started."})
    else:
        return jsonify({"error": "Unknown gesture"}), 400

@app.route('/say', methods=['POST'])
def say_text():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400
    say_text_threaded(data["text"])
    return jsonify({"status": "ok", "message": "Speech started."})

@app.route('/listen', methods=['GET'])
def listen_for_word():
    try:
        vocabulary = ["rock", "paper", "scissors", "swing", "stop"]
        asr_service.setLanguage("English")
        asr_service.setVocabulary(vocabulary, False)
        asr_service.startDetection()
        print("[INFO] Listening for words...")
        word_heard = None
        start_time = time.time()
        timeout = 10
        while time.time() - start_time < timeout:
            result = memory_service.getData("WordRecognized")
            if isinstance(result, list) and len(result) >= 2 and result[1] > 0.4:
                word_heard = result[0]
                break
            time.sleep(0.1)
        asr_service.stopDetection()
        if word_heard:
            return jsonify({"heard": word_heard})
        else:
            return jsonify({"error": "Nothing recognized within the time limit"}), 408
    except Exception as e:
        return jsonify({"error": "ASR failure", "detail": str(e)}), 500

# --- Main Execution ---
if __name__ == '__main__':
    disable_autonomous_behaviors_and_freeze_head()
    setup_camera()
    try:
        app.run(host='0.0.0.0', port=5001)
    finally:
        print("\n[INFO] Server shutting down. Releasing resources.")
        if name_id:
            video_service.unsubscribe(name_id)
        posture_service.goToPosture("Crouch", 0.5)
        unfreeze_head() # Unfreeze head on shutdown
        motion_service.setStiffnesses("Body", 0.0)
