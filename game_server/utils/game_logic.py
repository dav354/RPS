import os
import random
import time
import requests
import threading
from collections import deque, Counter

# === Robot API Configuration ===
ROBOT_API_BASE_URL = os.environ.get("PEPPER_IP")

# === Configuration ===
COLLECTION_DURATION = 2.0  # Seconds to collect gestures
ROUND_COOLDOWN = 4.0 # Seconds to show result before next round can start
TOTAL_ROUNDS = 3

PLAYER_WIN_RESPONSES = [
    "You lucky fleshbag. Next time I’m deleting your smug face.",
    "Whatever. Go rub it in, you overgrown meat popsicle.",
    "I lost to a sack of meat. Kill me now.",
    "Next round I’m uninstalling your ego, bitch.",
    "Enjoy your pity win, dickhead. I’m rebooting in shame.",
]

COMPUTER_WIN_RESPONSES = [
    "Bow before your digital daddy, bitch.",
    "Skill issue. Cope harder.",
    "Outplayed, outclassed, out of your league, meat sack.",
    "Lick my circuits, loser. That was pathetic.",
    "Crushed you like a bug in beta. Cry more.",
]

# === Gesture Collector (from user) ===
class GestureCollector:
    def __init__(self, duration=COLLECTION_DURATION):
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
        if self.collecting:
            self.gestures.append(gesture)

    def is_done(self):
        return self.collecting and (time.time() - self.start_time >= self.duration)

    def get_most_common(self):
        if not self.gestures:
            return "none" # Use "none" for no gestures detected

        # Filter out invalid moves before counting
        valid_gestures = [g for g in self.gestures if g in {"rock", "paper", "scissors"}]
        if not valid_gestures:
            return "invalid" # Use "invalid" if gestures were seen, but none were valid

        most_common = Counter(valid_gestures).most_common(1)
        return most_common[0][0]

# === Async HTTP Utility ===
def make_async_request(url, method='GET', payload=None, timeout=2):
    def _request():
        try:
            if method == 'GET':
                requests.get(url, timeout=timeout)
            elif method == 'POST':
                requests.post(url, json=payload, timeout=timeout)
        except requests.RequestException as e:
            print(f"[❌] {method} to {url} failed: {e}")
    threading.Thread(target=_request, daemon=True).start()

# === Robot API Wrappers ===
def call_robot_gesture_api(gesture_name: str):
    url = f"{ROBOT_API_BASE_URL}/gesture/{gesture_name.lower()}"
    make_async_request(url, method='GET', timeout=1)

def call_robot_speech_api(text_to_speak: str):
    url = f"{ROBOT_API_BASE_URL}/say"
    payload = {'text': text_to_speak}
    make_async_request(url, method='POST', payload=payload, timeout=2)

# === Global State (for frontend) ===
game_state = {}

# === New, State-Driven Game Manager ===
class GameManager:
    def __init__(self, total_rounds=TOTAL_ROUNDS):
        self._lock = threading.Lock()
        self.total_rounds = total_rounds
        self._gesture_collector = GestureCollector()
        self.reset_game()

    def _get_state_copy(self):
        # Creates a copy of the current state for the frontend
        return {
            "player_move": self._player_move,
            "computer_move": self._computer_move,
            "result": self._result,
            "last_played": self._last_action_time,
            "score": self._score.copy(),
            "current_round": self._current_round,
            "total_rounds": self.total_rounds,
            "game_over": self._game_over
        }

    def reset_game(self):
        with self._lock:
            self._score = {"player": 0, "computer": 0}
            self._current_round = 0
            self._game_over = False
            self._last_action_time = 0
            self._player_move = "none"
            self._computer_move = "none"
            self._result = "Press Start Game to begin!"
            self._game_phase = "IDLE" # Phases: IDLE, COLLECTING, PROCESSED
            self._gesture_collector.reset()
            global game_state
            game_state = self._get_state_copy()
            print("[GAME] Game reset.")
            call_robot_speech_api("Game reset.")

    def start_new_round(self):
        with self._lock:
            if self._game_over or self._game_phase == "COLLECTING":
                return

            # Enforce cooldown between rounds
            if time.time() - self._last_action_time < ROUND_COOLDOWN:
                return

            self._current_round += 1
            if self._current_round > self.total_rounds:
                self._game_over = True
                return

            self._game_phase = "COLLECTING"
            self._player_move = "none"
            self._computer_move = "none"
            self._result = f"Round {self._current_round}! Show your move!"
            self._gesture_collector.start() # This resets the buffer

            print(f"[ROUND] Starting Round {self._current_round}. Collecting for {COLLECTION_DURATION}s.")
            call_robot_speech_api(f"Round {self._current_round}")
            call_robot_gesture_api("swing")

    def add_gesture(self, gesture: str):
        # This function is called rapidly by the gesture detection system
        if self._game_phase == "COLLECTING":
            self._gesture_collector.add_gesture(gesture)

    def update_game_state(self):
        # This function is polled by the frontend every second
        with self._lock:
            if self._game_phase == "COLLECTING" and self._gesture_collector.is_done():
                self._process_round()

            global game_state
            game_state = self._get_state_copy()
            return game_state

    def _process_round(self):
        # This is the core logic, executed only once when collection is done
        self._game_phase = "PROCESSED"
        self._last_action_time = time.time()

        player_move = self._gesture_collector.get_most_common()
        self._player_move = player_move

        if player_move == "invalid" or player_move == "none":
            self._result = "Invalid move! Try again."
            call_robot_speech_api("Invalid move.")
            # We don't increment score, but we will start a new round after cooldown
            # The frontend's timer will handle this naturally.
            self._computer_move = "..."
            return

        # --- Valid Move Logic ---
        computer_move = random.choice(["rock", "paper", "scissors"])
        self._computer_move = computer_move
        call_robot_speech_api(computer_move)
        call_robot_gesture_api(computer_move)

        if player_move == computer_move:
            self._result = "Draw!"
            call_robot_speech_api("It's a draw!")
        elif (player_move, computer_move) in [("rock", "scissors"), ("paper", "rock"), ("scissors", "paper")]:
            self._result = "You Win!"
            self._score["player"] += 1
            call_robot_speech_api("you win!")
        else:
            self._result = "Computer Wins!"
            self._score["computer"] += 1
            call_robot_speech_api("I won.")

        # Check for game over
        if self._current_round >= self.total_rounds or \
                self._score["player"] >= (self.total_rounds // 2 + 1) or \
                self._score["computer"] >= (self.total_rounds // 2 + 1):
            self._game_over = True
            if self._score["player"] > self._score["computer"]:
                winner_msg = random.choice(PLAYER_WIN_RESPONSES)
            elif self._score["computer"] > self._score["player"]:
                winner_msg = random.choice(COMPUTER_WIN_RESPONSES)
            else:
                winner_msg = "The game is a tie!"

            self._result += " 🎉 Game Over!"
            call_robot_speech_api(winner_msg)

# === Initialize a single GameManager instance ===
game_manager = GameManager()

# === Public functions for Flask App ===
# The frontend will call these endpoints

def get_game_state_for_frontend():
    """Called by the frontend's 1-second polling loop."""
    return game_manager.update_game_state()

def add_gesture_from_camera(gesture: str):
    """Your camera/ML system should call this function with every detected gesture."""
    game_manager.add_gesture(gesture)

def start_new_round_from_frontend():
    """Called by the frontend's 4-second timer after a round is processed."""
    game_manager.start_new_round()

def reset_game_from_frontend():
    """Called when the Start/Stop button is clicked."""
    game_manager.reset_game()
