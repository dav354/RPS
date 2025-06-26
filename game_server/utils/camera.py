import cv2
import numpy as np
import time

MAX_RETRIES = 10
RETRY_DELAY = 5

def setup_camera(camera_source: str, max_retries=MAX_RETRIES, retry_delay=RETRY_DELAY):
    cap = cv2.VideoCapture(camera_source)
    retries = 0
    while not cap.isOpened() and retries < max_retries:
        print(f"[❌] Failed to open video source: {camera_source} (retry {retries+1}/{max_retries})")
        cap.release()
        cap = cv2.VideoCapture(camera_source)
        time.sleep(retry_delay)
        retries += 1
    if not cap.isOpened():
        print(f"[❌] Could not open camera after {max_retries} attempts!")
    return cap

def get_display_frame(cap, camera_source, max_retries=MAX_RETRIES, retry_delay=RETRY_DELAY):
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
