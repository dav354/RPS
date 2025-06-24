import os
import random
import time
import requests
import threading

# === Robot API Configuration ===
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

# <<< CHANGED: Helper function to run network calls in a thread >>>
def _run_in_thread(target, args=()):
    """Starts a function in a daemon thread, so it doesn't block."""
    thread = threading.Thread(target=target, args=args)
    thread.daemon = True # Allows main program to exit even if threads are running
    thread.start()

def _make_gesture_api_call(gesture_name: str):
    """The actual blocking HTTP call for gestures."""
    url = f"{ROBOT_API_BASE_URL}/gesture/{gesture_name.lower()}"
    try:
        response = requests.get(url, timeout=1) # 1-second timeout
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        print(f"[🤖 API] Successfully called gesture '{gesture_name}': {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"[❌🤖 API Error] Could not call gesture '{gesture_name}': {e}")

def call_robot_gesture_api(gesture_name: str):
    """
    Makes a NON-BLOCKING HTTP GET request to the robot's gesture API endpoint.
    """
    _run_in_thread(_make_gesture_api_call, (gesture_name,))


def _make_speech_api_call(text_to_speak: str):
    """The actual blocking HTTP call for speech."""
    url = f"{ROBOT_API_BASE_URL}/say"
    headers = {'Content-Type': 'application/json'}
    payload = {'text': text_to_speak}
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=2) # 2-second timeout for speech
        response.raise_for_status() # Raise an HTTPError for bad responses
        print(f"[🤖 API] Successfully called speech API with text: '{text_to_speak}': {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"[❌🤖 API Error] Could not call speech API with text '{text_to_speak}': {e}")

def call_robot_speech_api(text_to_speak: str):
    """
    Makes a NON-BLOCKING HTTP POST request to the robot's speech API endpoint.
    """
    _run_in_thread(_make_speech_api_call, (text_to_speak,))

# --- The rest of your game_logic.py remains the same ---

def prepare_round():
    call_robot_speech_api("Play!")
    call_robot_gesture_api("swing")

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
    call_robot_speech_api("Game reset. Let's play again!")


def play_round(player_move: str) -> dict:
    global _last_result_time, _score, _game_over

    now = time.time()

    if _game_over:
        game_state["result"] = "Game over. Please reset to play again."
        return game_state

    if now - _last_result_time < COOLDOWN:
        cooldown_remaining = COOLDOWN - (now - _last_result_time)
        game_state["result"] = f"Cooldown... wait {cooldown_remaining:.1f}s"
        return game_state

    if player_move not in {"rock", "paper", "scissors"}:
        game_state.update({
            "error": "Invalid move",
            "player_move": player_move,
            "computer_move": "none",
            "result": "Invalid move",
            "score": _score.copy(),
            "last_played": now
        })
        call_robot_speech_api("Invalid move. Please choose rock, paper, or scissors.")
        return game_state

    computer_move = random.choice(["rock", "paper", "scissors"])
    call_robot_speech_api(computer_move)
    call_robot_gesture_api(computer_move)

    if player_move == computer_move:
        result = "Draw"
        call_robot_speech_api("try again")
    elif (player_move, computer_move) in [
        ("rock", "scissors"),
        ("paper", "rock"),
        ("scissors", "paper")
    ]:
        result = "You Win!"
        _score["player"] += 1
        call_robot_speech_api("you win")
    else:
        result = "Computer Wins!"
        _score["computer"] += 1
        call_robot_speech_api("you lose")

    _last_result_time = now

    game_state.update({
        "player_move": player_move,
        "computer_move": computer_move,
        "result": result,
        "last_played": now,
        "score": _score.copy()
    })

    if _score["player"] == 2 or _score["computer"] == 2:
        game_over_message = " 🎉 Game Over!"
        game_state["result"] += game_over_message
        _game_over = True
        if _score["player"] == 2:
            call_robot_speech_api("Congratulations! You won the game!")
        else:
            call_robot_speech_api("I won the game! Better luck next time.")
        call_robot_speech_api(game_over_message)

    return game_state