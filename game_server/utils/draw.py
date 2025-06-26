# utils/draw.py
import cv2

HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
]


# MODIFIED: This function now expects 'landmarks' to be in pixel coordinates (e.g., int32)
# The 'width' and 'height' parameters are no longer used for scaling inside this function,
# but they might still be useful for context if you add bounds checks or other logic.
def draw_landmarks(frame, landmarks_pixel_coords, width, height): # Renamed 'landmarks' to 'landmarks_pixel_coords' for clarity
    if landmarks_pixel_coords is None: # Added a check in case no hand was detected
        return

    for i, lm in enumerate(landmarks_pixel_coords):
        x, y = int(lm[0]), int(lm[1]) # No multiplication by width/height here
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1) # Red dots

    for start_idx, end_idx in HAND_CONNECTIONS:
        if start_idx < len(landmarks_pixel_coords) and end_idx < len(landmarks_pixel_coords):
            x0, y0 = int(landmarks_pixel_coords[start_idx][0]), int(
                landmarks_pixel_coords[start_idx][1]
            )
            x1, y1 = int(landmarks_pixel_coords[end_idx][0]), int(
                landmarks_pixel_coords[end_idx][1]
            )
            cv2.line(frame, (x0, y0), (x1, y1), (0, 255, 0), 2) # Green linese