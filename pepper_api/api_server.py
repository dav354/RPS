#!/usr/bin/env python
# -*- coding: utf-8 -*-

import qi
import time
import cv2 
import numpy as np
from flask import Flask, Response, request, jsonify

app = Flask(__name__)

# Set up qi session to connect to the robot
session = qi.Session()
session.connect("tcp://127.0.0.1:9559")

# Get NAOqi services
video_service = session.service("ALVideoDevice")
motion_service = session.service("ALMotion")
posture_service = session.service("ALRobotPosture")
tts_service = session.service("ALTextToSpeech")
asr_service = session.service("ALSpeechRecognition")
memory_service = session.service("ALMemory")

# === Disable Autonomous Movements ===
def disable_autonomous_behaviors():
    try:
        awareness_service.stopAwareness()
        life_service.setState("disabled")
        motion_service.setStiffnesses("Body", 1.0)
        print("[🔧] Autonomous behaviors disabled.")
    except Exception as e:
        print("[⚠️] Error disabling autonomous life:", e

# === Camera Setup ===
camera_name = "flask_cam"
camera_index = 0  # top camera
resolution = 2    # 640x480
color_space = 13  # BGR
fps = 30

# Subscribe to the camera
name_id = video_service.subscribeCamera(camera_name, camera_index, resolution, color_space, fps)

def generate():
    while True:
        image = video_service.getImageRemote(name_id)
        if image is None:
            continue

        width = image[0]
        height = image[1]
        array = image[6]

        if not array:
            continue

        img = np.frombuffer(array, dtype=np.uint8).reshape((height, width, 3))
        ret, jpeg = cv2.imencode('.jpg', img)
        if not ret:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
        time.sleep(0.001)

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return "Camera stream is available at /video_feed"

# === Gestures ===

def go_to_neutral():
    posture_service.goToPosture("StandInit", 0.5)

def do_rock():
    motion_service.setStiffnesses("RArm", 1.0)
    motion_service.setAngles(["RShoulderPitch", "RElbowRoll", "RWristYaw", "RHand"],
                             [1.0, 0.5, 0.0, 0.0], 0.2)
    time.sleep(4)
    go_to_neutral()

def do_paper():
    motion_service.setStiffnesses("RArm", 1.0)
    # Vertical open hand, palm forward
    motion_service.setAngles([
        "RShoulderPitch", "RShoulderRoll", "RElbowRoll", "RWristYaw", "RHand"
    ], [
        0.8, 0.0, 1.0, -1.2, 1.0  # arm straight, wrist neutral, hand open
    ], 0.2)
    time.sleep(4)
    go_to_neutral()

def do_scissors():
    motion_service.setStiffnesses("RArm", 1.0)
    # Vertical open hand, palm forward
    motion_service.setAngles([
        "RShoulderPitch", "RShoulderRoll", "RElbowRoll", "RWristYaw", "RHand"
    ], [
        0.8, 0.0, 1.0, 0.0, 1.0  # arm straight, wrist neutral, hand open
    ], 0.2)
    time.sleep(4)
    go_to_neutral()


def do_swing():
    motion_service.setStiffnesses("RArm", 1.0)
    for _ in range(4):
        motion_service.setAngles(["RShoulderPitch", "RElbowRoll"], [0.4, 1.1], 0.3)
        time.sleep(0.25)
        motion_service.setAngles(["RShoulderPitch", "RElbowRoll"], [1.3, 0.4], 0.3)
        time.sleep(0.25)
    go_to_neutral()

@app.route('/gesture/<gesture_name>')
def do_gesture(gesture_name):
    gesture_name = gesture_name.lower()
    if gesture_name == "rock":
        do_rock()
    elif gesture_name == "paper":
        do_paper()
    elif gesture_name == "scissors":
        do_scissors()
    elif gesture_name == "swing":
        do_swing()
    else:
        return "Unknown gesture", 400
    return "Gesture performed: " + gesture_name

# === Text-to-Speech ===

@app.route('/say', methods=['POST'])
def say_text():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    text = data.get("text")

    if not text or not isinstance(text, basestring):
        return jsonify({"error": "Missing or invalid 'text' field"}), 400

    try:
        tts_service.say(text)
    except Exception as e:
        return jsonify({"error": "Text-to-speech failed", "detail": str(e)}), 500

    return jsonify({"status": "ok", "spoken": text})

@app.route('/listen', methods=['GET'])
def listen_for_word():
    try:
        vocabulary = ["rock", "paper", "scissors", "swing", "stop"]

        asr_service.setLanguage("English")
        asr_service.setVocabulary(vocabulary, False)

        asr_service.setMicrophonesAsInput()  # ✅ no arguments
        asr_service.startDetection()

        tts_service.say("Listening...")
        print("Listening...")

        word_heard = None
        start_time = time.time()
        timeout = 10  # seconds

        while time.time() - start_time < timeout:
            result = memory_service.getData("WordRecognized")
            if isinstance(result, list) and len(result) >= 2 and result[1] > 0.4:
                word_heard = result[0]
                break
            time.sleep(0.2)

        asr_service.stopDetection()

        if word_heard:
            return jsonify({"heard": word_heard})
        else:
            return jsonify({"error": "Nothing recognized"}), 408

    except Exception as e:
        return jsonify({"error": "ASR failure", "detail": str(e)}), 500


if __name__ == '__main__':
    disable_autonomous_behaviors()
    app.run(host='0.0.0.0', port=5001)
