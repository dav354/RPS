#!/usr/bin/env python2

import qi
import time
import cv2
import numpy as np
from flask import Flask, Response, request, jsonify
import threading
import atexit

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

# --- Camera Constants ---
CAMERA_INDEX = 0   # 0 = top, 1 = bottom
RESOLUTION = 2     # 2 = 640x480
COLOR_SPACE = 13   # 13 = BGR
FPS = 15           # Reduced FPS to lower load

# --- Optimized Camera Thread ---
class CameraThread(threading.Thread):
    def __init__(self):
        super(CameraThread, self).__init__()
        self.daemon = True
        self.name_id = None
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = True

    def run(self):
        print("[INFO] Starting camera thread...")
        try:
            self.name_id = video_service.subscribeCamera(
                "flask_cam_thread", CAMERA_INDEX, RESOLUTION, COLOR_SPACE, FPS
            )
            while self.running:
                image = video_service.getImageRemote(self.name_id)
                if image is None:
                    time.sleep(1.0 / FPS)
                    continue

                width, height, _, array = image[0], image[1], image[2], image[6]
                img = np.frombuffer(array, dtype=np.uint8).reshape((height, width, 3))
                ret, jpeg = cv2.imencode('.jpg', img)

                if ret:
                    with self.lock:
                        self.latest_frame = jpeg.tobytes()
                time.sleep(1.0 / FPS)
        except Exception as e:
            print("[ERROR] Camera thread failed:", e)
        finally:
            if self.name_id:
                video_service.unsubscribe(self.name_id)
            print("[INFO] Camera thread stopped.")

    def stop(self):
        self.running = False
        # No need to join a daemon thread, it will exit with the main program

    def get_frame(self):
        with self.lock:
            return self.latest_frame

# --- Event-Driven Speech Recognition ---
class SpeechRecognitionHandler(object):
    def __init__(self):
        self.word_heard = None
        self.event = threading.Event()
        self.subscriber = None
        try:
            self.subscriber = memory_service.subscriber("WordRecognized")
            self.connection = self.subscriber.signal.connect(self.on_word_recognized)
            print("[INFO] ASR event handler connected.")
        except Exception as e:
            print("[ERROR] Could not create ASR subscriber:", e)
            self.subscriber = None

    def on_word_recognized(self, value):
        if isinstance(value, list) and len(value) >= 2 and value[1] > 0.4:
            print("[INFO] Word recognized callback:", value[0])
            self.word_heard = value[0]
            self.event.set()

    def listen(self, vocabulary, timeout=10):
        if not self.subscriber:
            raise RuntimeError("ASR handler not connected.")
        self.word_heard = None
        self.event.clear()
        try:
            asr_service.setLanguage("English")
            asr_service.setVocabulary(vocabulary, False)
            asr_service.startDetection()
            print("[INFO] Listening for words (event-driven)...")
            triggered = self.event.wait(timeout)
            asr_service.stopDetection()
            return self.word_heard if triggered else None
        finally:
            asr_service.stopDetection()

    def disconnect(self):
        if self.subscriber and self.connection:
            try:
                self.subscriber.signal.disconnect(self.connection)
                print("[INFO] ASR event handler disconnected.")
            except Exception as e:
                print("[ERROR] Could not disconnect ASR subscriber:", e)

# --- Global Instances ---
camera_thread = CameraThread()
asr_handler = SpeechRecognitionHandler()

# --- Integrated Head Control Logic ---
def freeze_head():
    try:
        motion_service.setStiffnesses(["HeadYaw", "HeadPitch"], 1.0)
    except Exception as e:
        print("[ERROR] Could not freeze head:", e)

def unfreeze_head():
    try:
        motion_service.setStiffnesses(["HeadYaw", "HeadPitch"], 0.0)
    except Exception as e:
        print("[ERROR] Could not unfreeze head:", e)

# --- Autonomous Behavior Control ---
def disable_autonomous_behaviors():
    try:
        if awareness_service.isEnabled():
            awareness_service.setEnabled(False)
            print("[CONFIG] Basic Awareness disabled.")
        if life_service.getState() != "disabled":
            life_service.setState("disabled")
            print("[CONFIG] Autonomous Life disabled.")
        motion_service.setStiffnesses("Body", 1.0)
        print("[CONFIG] Body stiffness set to 1.0.")
        freeze_head()
        print("[CONFIG] Head is frozen.")
    except Exception as e:
        print("[WARN] Error during autonomous behavior disabling:", e)

# --- Video Stream Generation ---
def generate_video_stream():
    if not camera_thread.is_alive():
        return
    while True:
        frame = camera_thread.get_frame()
        if frame:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.01)

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return "Pepper API Server is running. Video at /video_feed"

# --- Threading Decorator ---
def run_in_thread(target_func):
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=target_func, args=args, kwargs=kwargs)
        thread.daemon = True
        thread.start()
    return wrapper

# --- Robot Actions (Non-Blocking) ---
@run_in_thread
def do_gesture(gesture_name):
    gesture_map = {
        "rock": ([1.0, 0.5, 0.0, 0.0], ["RShoulderPitch", "RElbowRoll", "RWristYaw", "RHand"]),
        "paper": ([0.8, 0.0, 1.0, -1.2, 1.0], ["RShoulderPitch", "RShoulderRoll", "RElbowRoll", "RWristYaw", "RHand"]),
        "scissors": ([0.8, 0.0, 1.0, 0.0, 1.0], ["RShoulderPitch", "RShoulderRoll", "RElbowRoll", "RWristYaw", "RHand"])
    }
    if gesture_name not in gesture_map: return

    print("[ACTION] Performing:", gesture_name)
    angles, joints = gesture_map[gesture_name]
    motion_service.setStiffnesses("RArm", 1.0)
    motion_service.setAngles(joints, angles, 0.2)
    time.sleep(GESTURE_HOLD_DURATION)
    posture_service.goToPosture("StandInit", 0.5)

@run_in_thread
def do_swing():
    print("[ACTION] Performing: swing")
    posture_service.goToPosture("StandInit", 0.5)
    motion_service.setStiffnesses("RArm", 1.0)
    for _ in range(2):
        motion_service.setAngles(["RShoulderPitch", "RElbowRoll"], [0.4, 1.1], 0.3)
        time.sleep(0.3)
        motion_service.setAngles(["RShoulderPitch", "RElbowRoll"], [1.3, 0.4], 0.3)
        time.sleep(0.3)

@run_in_thread
def say_text_threaded(text):
    try:
        tts_service.say(str(text))
    except Exception as e:
        print("[ERROR] TTS Error:", e)

# --- Flask Routes for Actions ---
@app.route('/gesture/<gesture_name>')
def perform_gesture(gesture_name):
    name = gesture_name.lower()
    if name in ["rock", "paper", "scissors"]:
        do_gesture(name)
        return jsonify({"status": "ok", "message": "Gesture " + name + " started."})
    elif name == "swing":
        do_swing()
        return jsonify({"status": "ok", "message": "Gesture swing started."})
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
def listen_for_word_route():
    try:
        vocabulary = ["rock", "paper", "scissors", "swing", "stop"]
        word_heard = asr_handler.listen(vocabulary, timeout=10)
        if word_heard:
            return jsonify({"heard": word_heard})
        else:
            return jsonify({"error": "Nothing recognized within the time limit"}), 408
    except Exception as e:
        return jsonify({"error": "ASR failure", "detail": str(e)}), 500

# --- Cleanup Function ---
def cleanup():
    print("\n[INFO] Server shutting down. Releasing resources...")
    camera_thread.stop()
    asr_handler.disconnect()
    posture_service.goToPosture("Crouch", 0.5)
    unfreeze_head()
    motion_service.setStiffnesses("Body", 0.0)
    print("[INFO] Resources released.")

# --- Main Execution ---
if __name__ == '__main__':
    disable_autonomous_behaviors()
    camera_thread.start()
    atexit.register(cleanup) # Register cleanup to be called on exit
    # For development, use Flask's built-in server:
    app.run(host='0.0.0.0', port=5001)
    # For production, use Gunicorn (see instructions).