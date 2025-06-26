import os
import random
import time
import requests
import threading

# === Robot API Configuration ===
ROBOT_API_BASE_URL = os.environ.get("PEPPER_IP")

# === Configuration ===
COOLDOWN = 3  # Seconds between rounds

# === Internal State ===
_last_result_time = 0
_score = {"player": 0, "computer": 0}
_game_over = False

# === External Game State ===
game_state = {
    "player_move": "none",
    "computer_move": "none",
    "result": "",
    "last_played": 0,
    "score": _score.copy()
}

# === Async HTTP Utility ===
def make_async_request(url, method='GET', payload=None, timeout=2):
    def _request():
        try:
            if method == 'GET':
                response = requests.get(url, timeout=timeout)
            elif method == 'POST':
                response = requests.post(url, json=payload, timeout=timeout)
            else:
                raise ValueError("Unsupported HTTP method")
            print(f"[✅] {method} to {url} succeeded: {response.status_code}")
        except requests.RequestException as e:
            print(f"[❌] {method} to {url} failed: {e}")
    threading.Thread(target=_request, daemon=True).start()

# === Robot Gesture API ===
def call_robot_gesture_api(gesture_name: str):
    url = f"{ROBOT_API_BASE_URL}/gesture/{gesture_name.lower()}"
    make_async_request(url, method='GET', timeout=1)

# === Robot Speech API ===
def call_robot_speech_api(text_to_speak: str):
    url = f"{ROBOT_API_BASE_URL}/say"
    payload = {'text': text_to_speak}
    make_async_request(url, method='POST', payload=payload, timeout=2)

# === Game Logic ===
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
