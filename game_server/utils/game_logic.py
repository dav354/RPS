import os
import random
import time
import requests
import threading

# === Robot API Configuration ===
ROBOT_API_BASE_URL = os.environ.get("PEPPER_IP")

# === Configuration ===
COOLDOWN = 3  # Seconds between actions/rounds
TOTAL_ROUNDS = 3 # Total rounds per game

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


# === Global Game State (managed by GameManager instance) ===
# This dictionary is the single source of truth for the frontend.
# It is ONLY written to by the GameManager instance in a thread-safe way.
game_state = {}

# === Async HTTP Utility ===
def make_async_request(url, method='GET', payload=None, timeout=2):
    """
    Sends an asynchronous HTTP request to the specified URL.
    The response is not awaited in this function.
    """
    def _request():
        try:
            if method == 'GET':
                response = requests.get(url, timeout=timeout)
            elif method == 'POST':
                response = requests.post(url, json=payload, timeout=timeout)
            else:
                raise ValueError("Unsupported HTTP method")
        except requests.RequestException as e:
            print(f"[❌] {method} to {url} failed: {e}")
    threading.Thread(target=_request, daemon=True).start()

# === Robot API Wrappers ===
def call_robot_gesture_api(gesture_name: str):
    """Triggers a gesture on the robot asynchronously."""
    url = f"{ROBOT_API_BASE_URL}/gesture/{gesture_name.lower()}"
    make_async_request(url, method='GET', timeout=1)

def call_robot_speech_api(text_to_speak: str):
    """Makes the robot speak asynchronously."""
    url = f"{ROBOT_API_BASE_URL}/say"
    payload = {'text': text_to_speak}
    make_async_request(url, method='POST', payload=payload, timeout=2)

# === Game Management Class ===
class GameManager:
    """
    Manages the game state in a thread-safe manner.
    This class is the single source of truth for all game logic.
    """
    def __init__(self, total_rounds=TOTAL_ROUNDS):
        self._lock = threading.Lock() # Lock to ensure thread safety
        self.total_rounds = total_rounds
        self.prepare_game() # Initialize all state variables

    def _update_global_state(self):
        """Copies the internal state to the global dictionary for the frontend."""
        global game_state
        game_state = {
            "player_move": self._player_move,
            "computer_move": self._computer_move,
            "result": self._result,
            "last_played": self._last_round_time,
            "score": self._score.copy(),
            "current_round": self._current_round,
            "total_rounds": self.total_rounds,
            "game_over": self._game_over
        }

    def prepare_game(self):
        """Prepares for a new game, resetting all internal state."""
        with self._lock:
            self._score = {"player": 0, "computer": 0}
            self._current_round = 0
            self._game_over = False
            self._last_round_time = 0
            self._is_awaiting_replay = False
            self._player_move = "none"
            self._computer_move = "none"
            self._result = "Press Start Game to begin!"
            self._update_global_state()
            print("[GAME] Game reset. Preparing for new game.")
            call_robot_speech_api("Game reset.")

    def start_new_round(self):
        """Initiates a new round. This function no longer checks for cooldown."""
        with self._lock:
            now = time.time()

            if self._game_over:
                return

            if self._current_round >= self.total_rounds:
                self._game_over = True
                self._result = "Game over."
                self._update_global_state()
                return

            self._is_awaiting_replay = False
            self._current_round += 1
            self._player_move = "none"
            self._computer_move = "none"
            self._result = f"Round {self._current_round}/{self.total_rounds}: Waiting for player..."
            self._last_round_time = now

            self._update_global_state()

            print(f"[ROUND] Starting Round {self._current_round}/{self.total_rounds}")
            call_robot_speech_api(f"Round {self._current_round}. Show your move!")
            call_robot_gesture_api("swing")

    def play_round(self, player_move: str):
        """
        Processes a single round, handling valid and invalid moves safely.
        This function is now the sole gatekeeper for the round cooldown.
        """
        with self._lock:
            now = time.time()

            if self._game_over:
                return

            # Cooldown check is now the first thing we do.
            # This prevents any move from being processed if the last action was too recent.
            if now - self._last_round_time < COOLDOWN:
                return

            if player_move not in {"rock", "paper", "scissors"}:
                if self._is_awaiting_replay:
                    return # Ignore repeated invalid moves

                self._is_awaiting_replay = True
                call_robot_speech_api("Invalid move. Let's try that again.")
                self._current_round -= 1
                self._player_move = player_move
                self._computer_move = "..."
                self._result = "Invalid move! Re-doing round..."
                self._last_round_time = now
                self._update_global_state()
                return

            # --- Valid Move Logic ---
            self._is_awaiting_replay = False # A valid move was received
            self._player_move = player_move
            self._computer_move = random.choice(["rock", "paper", "scissors"])

            call_robot_speech_api(self._computer_move)
            call_robot_gesture_api(self._computer_move)

            if self._player_move == self._computer_move:
                self._result = "Draw!"
                call_robot_speech_api("It's a draw!")
            elif (self._player_move, self._computer_move) in [("rock", "scissors"), ("paper", "rock"), ("scissors", "paper")]:
                self._result = "You Win!"
                self._score["player"] += 1
                call_robot_speech_api("you win!")
            else:
                self._result = "Computer Wins!"
                self._score["computer"] += 1
                call_robot_speech_api("I won.")

            self._last_round_time = now

            # Check for game over
            if self._current_round >= self.total_rounds or \
                    self._score["player"] >= (self.total_rounds // 2 + 1) or \
                    self._score["computer"] >= (self.total_rounds // 2 + 1):
                self._game_over = True
                final_winner = ""
                if self._score["player"] > self._score["computer"]:
                    final_winner = "Player"
                    call_robot_speech_api(random.choice(PLAYER_WIN_RESPONSES))
                elif self._score["computer"] > self._score["player"]:
                    final_winner = "Computer"
                    call_robot_speech_api(random.choice(COMPUTER_WIN_RESPONSES))
                else:
                    final_winner = "It's a tie!"
                    call_robot_speech_api("The game is a tie!")

                self._result += f" 🎉 Game Over! {final_winner} wins overall."
                print(f"[GAME OVER] {final_winner} wins.")

            self._update_global_state()

# === Initialize a single GameManager instance ===
game_manager = GameManager()

# === Public functions for Flask App ===
def prepare_round():
    """Wrapper to call GameManager's start_new_round."""
    game_manager.start_new_round()

def play_round(player_move: str):
    """Wrapper to call GameManager's play_round."""
    game_manager.play_round(player_move)

def reset_game():
    """Wrapper to call GameManager's prepare_game (for resetting)."""
    game_manager.prepare_game()
