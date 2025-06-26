import time
from collections import deque, Counter

class GestureCollector:
    def __init__(self, duration=2.0):
        self.duration = duration
        self.reset()

    def reset(self):
        self.gestures = deque()
        self.start_time = None
        self.collecting = False

    def start(self):
        self.reset()
        self.start_time = time.time()
        self.collecting = True

    def add_gesture(self, gesture):
        if self.collecting and gesture in ["rock", "paper", "scissors"]:
            self.gestures.append(gesture)

    def is_done(self):
        return self.collecting and (time.time() - self.start_time >= self.duration)

    def get_most_common(self):
        if not self.gestures:
            return "none"
        most_common = Counter(self.gestures).most_common(1)
        return most_common[0][0] if most_common else "none"
