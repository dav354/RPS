import cv2
import numpy as np
import time

def setup_camera(camera_source: str):
    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        print(f"[❌] Failed to open video source: {camera_source}")
    return cap

def get_display_frame(cap, camera_source, max_retries=5, retry_delay=2):
    """
    Returns (ok, frame, cap): 3rd return value is the possibly-new cap object!
    """
    retries = 0
    cap_valid = cap.isOpened()
    while not cap_valid and retries < max_retries:
        print(f"[⚠️] Camera not open. Attempting to fully reconnect... ({retries+1}/{max_retries})")
        cap.release()
        cap = cv2.VideoCapture(camera_source) 
        time.sleep(retry_delay)
        cap_valid = cap.isOpened()
        retries += 1
    if not cap_valid:
        frame = _error_frame("CAMERA UNAVAILABLE")
        return False, frame, cap

    ret, frame = cap.read()
    if not ret:
        print("[⚠️] Frame capture failed.")
        return False, _error_frame("FRAME CAPTURE FAIL"), cap

    return True, cv2.flip(frame, 1), cap

def _error_frame(message):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, message, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame
