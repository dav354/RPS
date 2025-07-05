import os
import random
import time
import requests
import threading
from collections import deque, Counter

# === Robot API Configuration ===
# Ensure the PEPPER_IP environment variable is set to your robot's IP address.
# Example: export PEPPER_IP="http://192.168.1.100"
ROBOT_API_BASE_URL = os.environ.get("PEPPER_IP")

# === Configuration ===
COLLECTION_DURATION = 2.0  # Seconds to collect gestures
ROUND_COOLDOWN = 6.0 # Seconds to show result before next round can start
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
    """Collects gestures over a short period and determines the most common valid one."""
    def __init__(self, duration=COLLECTION_DURATION):
        self.duration = duration
        self.reset()

    def reset(self):
        """Resets the collector's state."""
        self.gestures = deque()
        self.start_time = None
        self.collecting = False

    def start(self):
        """Starts a new collection window."""
        self.reset()
        self.start_time = time.time()
        self.collecting = True

    def add_gesture(self, gesture):
        """Adds a detected gesture to the buffer if collection is active."""
        if self.collecting:
            self.gestures.append(gesture)

    def is_done(self):
        """Checks if the collection duration has passed."""
        return self.collecting and (time.time() - self.start_time >= self.duration)

    def get_most_common(self):
        """
        Calculates the most common valid gesture from the collection period.
        Returns 'none' if no gestures were detected.
        Returns 'invalid' if gestures were detected, but none were valid.
        """
        if not self.gestures:
            return "none"

        # Filter out invalid moves before counting
        valid_gestures = [g for g in self.gestures if g in {"rock", "paper", "scissors"}]
        if not valid_gestures:
            return "invalid"

        most_common = Counter(valid_gestures).most_common(1)
        return most_common[0][0]

# === Async HTTP Utility ===
def make_async_request(url, method='GET', payload=None, timeout=2):
    """Sends an HTTP request in a separate thread to avoid blocking the main game loop."""
    def _request():
        if not ROBOT_API_BASE_URL:
            # Silently fail if the robot IP is not configured
            # print("[⚠️] ROBOT_API_BASE_URL not set. Skipping robot command.")
            return
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
    """Tells the robot to perform a gesture."""
    url = f"{ROBOT_API_BASE_URL}/gesture/{gesture_name.lower()}"
    make_async_request(url, method='GET', timeout=1)

def call_robot_speech_api(text_to_speak: str):
    """Tells the robot to say something."""
    url = f"{ROBOT_API_BASE_URL}/say"
    payload = {'text': text_to_speak}
    make_async_request(url, method='POST', payload=payload, timeout=2)

# === Global State (for frontend) ===
# This dictionary holds the game state that the frontend will poll for updates.
game_state = {}

# === State-Driven Game Manager ===
class GameManager:
    """Manages the entire game lifecycle, state, and logic."""
    def __init__(self, total_rounds=TOTAL_ROUNDS):
        self._lock = threading.Lock()
        self.total_rounds = total_rounds
        self._gesture_collector = GestureCollector()
        self.reset_game()

    def _get_state_copy(self):
        """Creates a thread-safe copy of the current state for the frontend."""
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
        """Resets the game to its initial state."""
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
            call_robot_speech_api("Game reset. Let's play rock paper scissors.")

    def start_new_round(self):
        """Starts a new round if the game is not over and the cooldown has passed."""
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
            self._gesture_collector.start()

            print(f"[ROUND] Starting Round {self._current_round}. Collecting for {COLLECTION_DURATION}s.")
            call_robot_speech_api(f"Round {self._current_round}")
            call_robot_gesture_api("swing")

    def add_gesture(self, gesture: str):
        """Adds a gesture from the camera/ML system to the collector."""
        if self._game_phase == "COLLECTING":
            self._gesture_collector.add_gesture(gesture)

    def update_game_state(self):
        """
        This function is polled by the frontend.
        It checks if the gesture collection period is over and processes the round if it is.
        """
        with self._lock:
            if self._game_phase == "COLLECTING" and self._gesture_collector.is_done():
                self._process_round()

            global game_state
            game_state = self._get_state_copy()
            return game_state

    def _process_round(self):
        """
        Contains the core game logic. Determines winner or handles invalid moves.
        This is executed only once when gesture collection is done.
        """

        self._game_phase = "PROCESSED"
        self._last_action_time = time.time()

        player_move = self._gesture_collector.get_most_common()
        self._player_move = player_move

        # --- FIX: Invalid Move Logic ---
        if player_move == "invalid" or player_move == "none":
            self._result = "Invalid move! Let's try that round again."
            call_robot_speech_api("Invalid move. Let's try again.")
            self._computer_move = "..."

            # Decrement the round counter. The frontend's cooldown timer will trigger
            # start_new_round(), which will increment it back to the same number,
            # effectively replaying the current round.
            if self._current_round > 0:
                self._current_round -= 1
            return # Stop further processing for this invalid round

        # --- Valid Move Logic ---
        computer_move = random.choice(["rock", "paper", "scissors"])
        self._computer_move = computer_move
        call_robot_gesture_api(computer_move)
        call_robot_speech_api(computer_move)




        if player_move == computer_move:
            self._result = "Draw!"
            call_robot_speech_api("It's a draw!")
        elif (player_move, computer_move) in [("rock", "scissors"), ("paper", "rock"), ("scissors", "paper")]:
            self._result = "You Win!"
            self._score["player"] += 1
            call_robot_speech_api("You Win!")
        else:
            self._result = "Computer Wins!"
            self._score["computer"] += 1
            call_robot_speech_api("I Won!")

        # Check for game over condition
        if self._current_round >= self.total_rounds or \
                self._score["player"] >= (self.total_rounds // 2 + 1) or \
                self._score["computer"] >= (self.total_rounds // 2 + 1):
            self._game_over = True
            final_msg = ""
            if self._score["player"] > self._score["computer"]:
                final_msg = random.choice(PLAYER_WIN_RESPONSES)
            elif self._score["computer"] > self._score["player"]:
                final_msg =random.choice(COMPUTER_WIN_RESPONSES)
            else:
                final_msg = "Game over. It's a tie. How boring."

            self._result = " 🎉 Game Over!"
            call_robot_speech_api(final_msg)


# === Initialize a single GameManager instance ===
game_manager = GameManager()

# === Public functions for a web framework like Flask ===
# The frontend would call these endpoints.

def get_game_state_for_frontend():
    """Called by the frontend's polling loop (e.g., every second)."""
    return game_manager.update_game_state()

def add_gesture_from_camera(gesture: str):
    """Your camera/ML system should call this function with every detected gesture."""
    game_manager.add_gesture(gesture)

def start_new_round_from_frontend():
    """
    Called by the frontend's timer after a round is processed (e.g., after ROUND_COOLDOWN).
    """
    game_manager.start_new_round()

def reset_game_from_frontend():
    """Called when a Start/Reset button is clicked in the UI."""
    game_manager.reset_game()

# Example of how you might run this in a simple loop for testing without a frontend
if __name__ == '__main__':
    print("--- Rock-Paper-Scissors Game Logic Test ---")
    print("This test simulates the interaction between the game logic and a frontend.")

    # Reset game
    reset_game_from_frontend()
    time.sleep(1)

    # Start the first round
    start_new_round_from_frontend()
    print(f"Frontend State: {get_game_state_for_frontend()}")

    # Simulate an invalid move
    print("\n--- SIMULATING INVALID ROUND ---")
    add_gesture_from_camera("wave") # an invalid gesture
    add_gesture_from_camera("point") # another invalid gesture

    # Wait for collection to finish
    time.sleep(COLLECTION_DURATION + 0.1)

    # Poll for state update, which will trigger processing
    state = get_game_state_for_frontend()
    print(f"Frontend State after invalid move: {state}")
    assert state["result"] == "Invalid move! Let's try that round again."
    assert state["current_round"] == 0 # Round was decremented back to 0

    # Wait for cooldown
    print(f"\nWaiting for {ROUND_COOLDOWN}s cooldown...")
    time.sleep(ROUND_COOLDOWN)

    # Frontend tries to start the next round
    print("\n--- REPLAYING THE ROUND ---")
    start_new_round_from_frontend()
    state = get_game_state_for_frontend()
    print(f"Frontend State at start of replayed round: {state}")
    assert state["current_round"] == 1 # Round is now correctly round 1 again

    # Simulate a valid move
    add_gesture_from_camera("rock")
    time.sleep(COLLECTION_DURATION + 0.1)

    state = get_game_state_for_frontend()
    print(f"Frontend State after valid move: {state}")
    assert state["current_round"] == 1
    assert state["result"] in ["You Win!", "Computer Wins!", "Draw!"]

    print("\n--- TEST COMPLETE ---")
