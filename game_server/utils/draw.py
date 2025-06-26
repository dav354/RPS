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


def draw_landmarks(frame, landmarks, width, height):
    for i, lm in enumerate(landmarks):
        x, y = int(lm[0] * width), int(lm[1] * height)
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

    for start_idx, end_idx in HAND_CONNECTIONS:
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            x0, y0 = int(landmarks[start_idx][0] * width), int(
                landmarks[start_idx][1] * height
            )
            x1, y1 = int(landmarks[end_idx][0] * width), int(
                landmarks[end_idx][1] * height
            )
            cv2.line(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
