#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import qi
import time
import cv2
import numpy as np
from flask import Flask, Response, request, jsonify
import threading
import logging
from collections import deque

# --- 1. Centralized Logging Service ---
# This class captures log messages into a memory buffer, allowing them
# to be viewed via a web endpoint.
class LogHandler:
    def __init__(self, max_logs=200):
        # A deque is a double-ended queue, efficient for adding and removing items.
        self.log_deque = deque(maxlen=max_logs)
        self.formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    def setup(self):
        """Configures the root logger to use our custom handler and a console handler."""
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)

        # Custom handler to store logs in our deque
        deque_handler = logging.Handler()
        deque_handler.setFormatter(self.formatter)
        deque_handler.emit = lambda record: self.log_deque.append(self.formatter.format(record))
        root_logger.addHandler(deque_handler)

        # Console handler to also print logs to the terminal
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(self.formatter)
        root_logger.addHandler(console_handler)

    def get_logs(self):
        """Returns all stored logs as a list."""
        return list(self.log_deque)

# --- 2. Centralized Video Stream Service ---
# This class manages the camera connection and frame grabbing in a single
# background thread, ensuring the robot's camera is only subscribed to once.
class VideoStreamer:
    def __init__(self, video_service):
        self.video_service = video_service
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.is_running = False
        self.name_id = None
        self.camera_name = "flask_cam_service"

    def start(self):
        """Subscribes to the camera and starts the background frame-grabbing thread."""
        if self.is_running:
            logging.warning("Video streamer is already running.")
            return

        camera_index = 0  # 0 = top camera
        resolution = 1    # 1 = 640x480
        color_space = 13  # 13 = BGR
        fps = 15          # Increased FPS slightly for smoother video

        try:
            self.name_id = self.video_service.subscribeCamera(
                self.camera_name, camera_index, resolution, color_space, fps
            )
            self.is_running = True
            thread = threading.Thread(target=self._camera_loop)
            thread.daemon = True
            thread.start()
            logging.info("Video streamer started successfully. Camera subscribed with ID: %s", self.name_id)
        except Exception as e:
            logging.error("Failed to subscribe to camera: %s", e, exc_info=True)
            self.is_running = False

    def _camera_loop(self):
        """The core loop that continuously fetches frames from the robot."""
        while self.is_running:
            try:
                image = self.video_service.getImageRemote(self.name_id)
                if image is None:
                    # If no image, wait a bit before trying again
                    time.sleep(0.01)
                    continue

                width, height, _, array = image[0], image[1], image[2], image[6]
                img = np.frombuffer(array, dtype=np.uint8).reshape((height, width, 3))
                ret, jpeg = cv2.imencode('.jpg', img)

                if ret:
                    with self.frame_lock:
                        self.latest_frame = jpeg.tobytes()

            except Exception as e:
                logging.warning("Camera loop error: %s", e)
                # If an error occurs, wait before retrying to avoid spamming logs
                time.sleep(1.0)
        logging.info("Camera loop has stopped.")

    def stop(self):
        """Stops the camera loop and unsubscribes from the video service."""
        logging.info("Stopping video streamer...")
        self.is_running = False
        if self.name_id:
            try:
                self.video_service.unsubscribe(self.name_id)
                logging.info("Camera unsubscribed successfully.")
            except Exception as e:
                logging.error("Failed to unsubscribe from camera: %s", e, exc_info=True)
        self.name_id = None

    def generate_frames(self):
        """A generator function that yields frames for the Flask response."""
        while True:
            with self.frame_lock:
                frame = self.latest_frame
            if frame:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            # Small delay to prevent the loop from consuming too much CPU
            time.sleep(0.05)

# --- 3. Application Setup ---
app = Flask(__name__)
log_handler = LogHandler()
log_handler.setup()

# --- NAOqi Service Connection ---
session = None
video_service = None
motion_service = None
posture_service = None
tts_service = None
asr_service = None
memory_service = None
life_service = None
awareness_service = None
video_streamer = None

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
    # Instantiate the video streamer with the connected service
    video_streamer = VideoStreamer(video_service)
    logging.info("Successfully connected to NAOqi services.")
except Exception as e:
    logging.critical("Fatal error connecting to NAOqi services: %s", e, exc_info=True)
    # If connection fails, we can't do anything else.
    exit(1)

GESTURE_HOLD_DURATION = 4

# --- Utility Functions and Decorators ---
def run_in_thread(target_func):
    """Decorator to run a function in a separate daemon thread."""
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=target_func, args=args, kwargs=kwargs)
        thread.daemon = True
        thread.start()
    return wrapper

# --- Robot Control Functions ---
def freeze_head():
    logging.info("Freezing head position.")
    try:
        motion_service.setStiffnesses(["HeadYaw", "HeadPitch"], 1.0)
    except Exception as e:
        logging.error("Could not freeze head: %s", e, exc_info=True)

def unfreeze_head():
    logging.info("Unfreezing head position.")
    try:
        motion_service.setStiffnesses(["HeadYaw", "HeadPitch"], 0.0)
    except Exception as e:
        logging.error("Could not unfreeze head: %s", e, exc_info=True)

def disable_autonomous_behaviors_and_freeze_head():
    try:
        # CORRECTED: Use isEnabled() and setEnabled() for ALBasicAwareness
        if awareness_service.isEnabled():
            awareness_service.setEnabled(False)
            logging.info("Basic Awareness disabled.")

        if life_service.getState() != "disabled":
            life_service.setState("disabled")
            logging.info("Autonomous Life disabled.")

        motion_service.setStiffnesses("Body", 1.0)
        logging.info("Body stiffness set to 1.0.")
        freeze_head()

    except Exception as e:
        logging.warning("Error during autonomous behavior disabling: %s", e, exc_info=True)

# --- Robot Actions ---
@run_in_thread
def do_rock():
    logging.info("Performing gesture: rock")
    motion_service.setStiffnesses("RArm", 1.0)
    motion_service.setAngles(["RShoulderPitch", "RElbowRoll", "RWristYaw", "RHand"], [1.0, 0.5, 0.0, 0.0], 0.2)
    time.sleep(GESTURE_HOLD_DURATION)
    posture_service.goToPosture("StandInit", 0.5)

@run_in_thread
def do_paper():
    logging.info("Performing gesture: paper")
    motion_service.setStiffnesses("RArm", 1.0)
    motion_service.setAngles(["RShoulderPitch", "RShoulderRoll", "RElbowRoll", "RWristYaw", "RHand"], [0.8, 0.0, 1.0, -1.2, 1.0], 0.2)
    time.sleep(GESTURE_HOLD_DURATION)
    posture_service.goToPosture("StandInit", 0.5)

@run_in_thread
def do_scissors():
    logging.info("Performing gesture: scissors")
    motion_service.setStiffnesses("RArm", 1.0)
    motion_service.setAngles(["RShoulderPitch", "RShoulderRoll", "RElbowRoll", "RWristYaw", "RHand"], [0.8, 0.0, 1.0, 0.0, 1.0], 0.2)
    time.sleep(GESTURE_HOLD_DURATION)
    posture_service.goToPosture("StandInit", 0.5)

@run_in_thread
def do_swing():
    logging.info("Performing gesture: swing")
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
        logging.info("Saying: '%s'", text)
        tts_service.say(str(text))
    except Exception as e:
        logging.error("Text-to-speech error: %s", e, exc_info=True)

# --- 4. Flask API Routes ---
@app.route('/')
def index():
    return "Pepper API Server is running. Video at /video_feed, Logs at /logs"

@app.route('/logs')
def get_server_logs():
    """New endpoint to view captured logs."""
    return jsonify(log_handler.get_logs())

@app.route('/video_feed')
def video_feed():
    """Serves the video stream using the centralized streamer."""
    if not video_streamer or not video_streamer.is_running:
        return "Video streamer is not running.", 503
    return Response(video_streamer.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/gesture/<gesture_name>')
def perform_gesture(gesture_name):
    gesture_map = {
        "rock": do_rock, "paper": do_paper,
        "scissors": do_scissors, "swing": do_swing
    }
    func = gesture_map.get(gesture_name.lower())
    if func:
        func()
        return jsonify({"status": "ok", "message": "Gesture " + gesture_name + " started."})
    else:
        logging.warning("Unknown gesture requested: %s", gesture_name)
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
        logging.info("Listening for words: %s", vocabulary)
        word_heard = None
        start_time = time.time()
        timeout = 10
        while time.time() - start_time < timeout:
            result = memory_service.getData("WordRecognized")
            if isinstance(result, list) and len(result) >= 2 and result[1] > 0.4:
                word_heard = result[0]
                logging.info("Word recognized: '%s' with confidence %s", result[0], result[1])
                break
            time.sleep(0.1)
        asr_service.stopDetection()
        if word_heard:
            return jsonify({"heard": word_heard})
        else:
            logging.info("Listening timed out. No word recognized.")
            return jsonify({"error": "Nothing recognized within the time limit"}), 408
    except Exception as e:
        logging.error("Speech recognition failure: %s", e, exc_info=True)
        return jsonify({"error": "ASR failure", "detail": str(e)}), 500

# --- 5. Main Execution ---
if __name__ == '__main__':
    disable_autonomous_behaviors_and_freeze_head()
    if video_streamer:
        video_streamer.start()

    try:
        # Using threaded=True is important for handling multiple requests
        app.run(host='0.0.0.0', port=5001, threaded=True)
    finally:
        logging.info("Server shutting down. Releasing resources.")
        if video_streamer:
            video_streamer.stop()
        if posture_service:
            posture_service.goToPosture("Crouch", 0.5)
        unfreeze_head()
        if motion_service:
            motion_service.setStiffnesses("Body", 0.0)
        logging.info("All resources released. Goodbye.")