import cv2
import numpy as np
import time

def setup_camera(camera_source: str):
    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        print(f"[❌] Failed to open video source: {camera_source}")
    return cap

def get_display_frame(cap, camera_source, max_retries=10, retry_delay=5):
    retries = 0
    while not cap.isOpened() and retries < max_retries:
        print(f"[⚠️] Camera not open. Attempting to reconnect... ({retries+1}/{max_retries})")
        cap.release()
        cap.open(camera_source)
        time.sleep(retry_delay)
        retries += 1
    if not cap.isOpened():
        frame = _error_frame("CAMERA UNAVAILABLE")
        return False, frame

    ret, frame = cap.read()
    if not ret:
        print("[⚠️] Frame capture failed.")
        return False, _error_frame("FRAME CAPTURE FAIL")

    return True, cv2.flip(frame, 1)

def _error_frame(message):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, message, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame
