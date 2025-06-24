import os
import random
import time
import requests # Import the requests library for making HTTP calls

# === Robot API Configuration ===
# IMPORTANT: Change this to your robot's actual API base URL
ROBOT_API_BASE_URL = os.environ.get("PEPPER_IP") # Assuming robot API is on this address

# === Konfiguration ===
COOLDOWN = 3  # Sekunden zwischen Runden

# === Interner Zustand ===
_last_result_time = 0
_score = {"player": 0, "computer": 0}
_game_over = False

# === Spielstatus, extern abrufbar ===
game_state = {
    "player_move": "none",
    "computer_move": "none",
    "result": "",
    "last_played": 0,
    "score": _score.copy()
}

def call_robot_gesture_api(gesture_name: str):
    """
    Makes an HTTP GET request to the robot's gesture API endpoint.
    This function is now part of your game logic file.
    """
    url = f"{ROBOT_API_BASE_URL}/gesture/{gesture_name.lower()}"
    try:
        response = requests.get(url, timeout=1) # 5-second timeout
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        print(f"[🤖 API] Successfully called gesture '{gesture_name}': {response.text}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"[❌🤖 API Error] Could not call gesture '{gesture_name}': {e}")
        return False

def reset_game():
    global _score, _game_over, _last_result_time
    _score = {"player": 0, "computer": 0}
    _game_over = False
    _last_result_time = 0
    game_state.update({
        "player_move": "none",
        "computer_move": "none",
        "result": "",
        "last_played": 0,
        "score": _score.copy()
    })


def play_round(player_move: str) -> dict:
    global _last_result_time, _score, _game_over

    now = time.time()

    if _game_over:
        game_state["result"] = "Game over. Please reset to play again."
        return game_state

    if now - _last_result_time < COOLDOWN:
        game_state["result"] = f"Cooldown... wait {COOLDOWN - (now - _last_result_time):.1f}s"
        return game_state

    if player_move not in {"rock", "paper", "scissors"}:
        return {
            "error": "Invalid move",
            "player_move": player_move,
            "computer_move": "none",
            "result": "Invalid move",
            "score": _score.copy(),
            "last_played": now
        }

    computer_move = random.choice(["rock", "paper", "scissors"])

    if player_move == computer_move:
        result = "Draw"
    elif (player_move, computer_move) in [
        ("rock", "scissors"),
        ("paper", "rock"),
        ("scissors", "paper")
    ]:
        result = "You Win!"
        _score["player"] += 1
    else:
        result = "Computer Wins!"
        _score["computer"] += 1

    _last_result_time = now

    # --- API call for the robot's gesture based on computer_move ---
    call_robot_gesture_api(computer_move)
    # -------------------------------------------------------------

    game_state.update({
        "player_move": player_move,
        "computer_move": computer_move,
        "result": result,
        "last_played": now,
        "score": _score.copy()
    })

    # Siegbedingung prüfen
    if _score["player"] == 2 or _score["computer"] == 2:
        game_state["result"] += " 🎉 Game Over!"
        _game_over = True

    return game_state

# --- Example of how to use play_round and see the API call in action ---
if __name__ == "__main__":
    print("Starting a test game round...")
    # This will simulate a player move and trigger the robot gesture API call
    # You need your robot API running at ROBOT_API_BASE_URL (e.g., http://localhost:5000)
    # and the /gesture/<gesture_name> endpoint correctly configured.
    current_game_state = play_round("rock")
    print(f"Game State after first round: {current_game_state}")

    # Wait for cooldown
    time.sleep(COOLDOWN)

    current_game_state = play_round("paper")
    print(f"Game State after second round: {current_game_state}")

    # You could also call the 'swing' gesture directly from wherever you 'start' a round
    # For example, if you have a function that initiates the round from an external call:
    # def start_new_game_round():
    #     print("New game round initiated!")
    #     call_robot_gesture_api("swing")
    #     reset_game() # Or whatever game setup you do
    #
    # start_new_game_round()