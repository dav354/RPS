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
# This will be updated by the GameManager instance
game_state = {
    "player_move": "none",
    "computer_move": "none",
    "result": "",
    "last_played": 0,
    "score": {"player": 0, "computer": 0},
    "current_round": 0,
    "total_rounds": TOTAL_ROUNDS,
    "game_over": False
}

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
            # print(f"[✅] {method} to {url} succeeded: {response.status_code}")
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
    def __init__(self, total_rounds=TOTAL_ROUNDS):
        self.total_rounds = total_rounds
        self._score = {"player": 0, "computer": 0}
        self._current_round = 0
        self._game_over = False
        self._last_round_time = 0 # To manage cooldown between rounds
        self._update_global_game_state() # Initialize global state

    def _update_global_game_state(self):
        """Updates the global game_state dictionary from instance attributes."""
        global game_state
        game_state.update({
            "player_move": game_state.get("player_move", "none"), # Keep previous move for display
            "computer_move": game_state.get("computer_move", "none"), # Keep previous move for display
            "result": game_state.get("result", ""), # Keep previous result for display
            "last_played": self._last_round_time,
            "score": self._score.copy(),
            "current_round": self._current_round,
            "total_rounds": self.total_rounds,
            "game_over": self._game_over
        })

    def prepare_game(self):
        """Prepares for a new game, resetting scores and state."""
        self._score = {"player": 0, "computer": 0}
        self._current_round = 0
        self._game_over = False
        self._last_round_time = 0
        self._update_global_game_state()
        print("[GAME] Game reset. Preparing for new game.")
        call_robot_speech_api("Game reset.")

    def start_new_round(self):
        """Initiates a new round if the game is not over and cooldown allows."""
        now = time.time()

        if self._game_over:
            game_state["result"] = "Game over. Please reset to play again."
            self._update_global_game_state()
            return False # Cannot start round

        if now - self._last_round_time < COOLDOWN:
            cooldown_remaining = COOLDOWN - (now - self._last_round_time)
            game_state["result"] = f"Cooldown... wait {cooldown_remaining:.1f}s before next round."
            self._update_global_game_state()
            return False # Cannot start round

        if self._current_round >= self.total_rounds:
            # This handles cases where _game_over might not have been caught
            # e.g., if a new round is requested immediately after the last round finished.
            self._game_over = True
            game_state["result"] = "Game over."
            self._update_global_game_state()
            return False

        self._current_round += 1
        print(f"[ROUND] Starting Round {self._current_round}/{self.total_rounds}")
        call_robot_gesture_api("swing") # Robot swings to signal start of round
        game_state.update({ # Reset round-specific state
            "player_move": "none",
            "computer_move": "none",
            "result": f"Round {self._current_round}/{self.total_rounds}: Waiting for player...",
            "last_played": now
        })
        self._update_global_game_state()
        return True # Round started successfully

    def play_round(self, player_move: str) -> dict:
        """
        Processes a single round of Rock, Paper, Scissors.
        Assumes start_new_round has already been called for the current round.
        """
        now = time.time()

        # Re-check for game over or cooldown, though `start_new_round` should prevent most of this
        if self._game_over:
            game_state["result"] = "Game over."
            self._update_global_game_state()
            return game_state

        if now - self._last_round_time < COOLDOWN:
            cooldown_remaining = COOLDOWN - (now - self._last_round_time)
            game_state["result"] = f"Still in cooldown from previous round... wait {cooldown_remaining:.1f}s"
            self._update_global_game_state()
            return game_state

        if player_move not in {"rock", "paper", "scissors"}:
            game_state.update({
                "error": "Invalid move",
                "player_move": player_move,
                "computer_move": "none",
                "result": "Invalid move. Please choose rock, paper, or scissors.",
                "score": self._score.copy(),
                "last_played": now
            })
            call_robot_speech_api("Invalid move.")
            self._update_global_game_state()
            return game_state

        computer_move = random.choice(["rock", "paper", "scissors"])
        call_robot_speech_api(computer_move) # Announce robot's move
        call_robot_gesture_api(computer_move) # Perform robot's move

        result_message = ""
        if player_move == computer_move:
            result_message = "Draw!"
            call_robot_speech_api("It's a draw!")
        elif (player_move, computer_move) in [
            ("rock", "scissors"),
            ("paper", "rock"),
            ("scissors", "paper")
        ]:
            result_message = "You Win!"
            self._score["player"] += 1
            call_robot_speech_api("you win!")
        else:
            result_message = "Computer Wins!"
            self._score["computer"] += 1
            call_robot_speech_api("I won.")

        self._last_round_time = now
        game_state.update({
            "player_move": player_move,
            "computer_move": computer_move,
            "result": result_message,
            "last_played": now,
            "score": self._score.copy()
        })
        self._update_global_game_state()

        # Check for game over conditions after updating score for the round
        if self._current_round == self.total_rounds or \
                self._score["player"] == (self.total_rounds // 2 + 1) or \
                self._score["computer"] == (self.total_rounds // 2 + 1):
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

            game_state["result"] += f" 🎉 Game Over! {final_winner} wins overall."
            print(f"[GAME OVER] {final_winner} wins.")
            self._update_global_game_state()

        return game_state

# === Initialize the GameManager ===
game_manager = GameManager(total_rounds=TOTAL_ROUNDS)

# === Public functions for Flask App ===
def prepare_round():
    """Wrapper to call GameManager's start_new_round."""
    return game_manager.start_new_round()

def play_round(player_move: str) -> dict:
    """Wrapper to call GameManager's play_round."""
    return game_manager.play_round(player_move)

def reset_game():
    """Wrapper to call GameManager's prepare_game (for resetting)."""
    game_manager.prepare_game()