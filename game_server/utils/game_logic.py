# utils/game_logic.py
import os
import random
import time
import requests
import threading

# === Config ===
ROBOT_API_BASE_URL = f"http://{os.environ.get('PEPPER_IP')}:5001" if os.environ.get("PEPPER_IP") else ""
COOLDOWN = 3
TOTAL_ROUNDS = 3

def make_async_request(url, method='GET', payload=None, timeout=2):
    def _request():
        try:
            if method == 'GET':
                requests.get(url, timeout=timeout)
            elif method == 'POST':
                requests.post(url, json=payload, timeout=timeout)
        except requests.RequestException as e:
            print(f"[❌] Request failed: {e}")
    if ROBOT_API_BASE_URL: # Only try to make requests if IP is set
        threading.Thread(target=_request, daemon=True).start()

def call_robot_gesture_api(gesture: str):
    make_async_request(f"{ROBOT_API_BASE_URL}/gesture/{gesture.lower()}")

def call_robot_speech_api(text: str):
    make_async_request(f"{ROBOT_API_BASE_URL}/say", method='POST', payload={'text': text})

# === GameManager ===
class GameManager:
    def __init__(self, total_rounds=TOTAL_ROUNDS):
        self.total_rounds = total_rounds
        self.reset()

    def reset(self):
        self.round = 0
        self.score = {"player": 0, "computer": 0}
        self.last_time = 0
        self.over = False
        self.state = {
            "player_move": "none",
            "computer_move": "none",
            "result": "Press 'Start Game' to begin!",
            "score": self.score.copy(),
            "round": self.round,
            "game_over": self.over
        }
        call_robot_speech_api("Let's play a new game.")
        print("[GAME] Reset")

    def _cooldown_remaining(self):
        return max(0, COOLDOWN - (time.time() - self.last_time))

    def _set_result(self, result, update_state=True):
        self.state["result"] = result
        if update_state:
            self._update_state()

    def _update_state(self):
        self.state.update({
            "score": self.score.copy(),
            "round": self.round,
            "game_over": self.over,
        })

    def start_round(self):
        if self.over:
            return False

        self.round += 1
        self.last_time = time.time()
        self.state.update({
            "player_move": "none",
            "computer_move": "none",
            "result": f"Round {self.round}/{self.total_rounds}: Show your move!"
        })
        self._update_state()

        call_robot_speech_api(f"Round {self.round}. Rock, paper, scissors, shoot!")
        call_robot_gesture_api("swing")
        print(f"[ROUND] Round {self.round}/{self.total_rounds} started")
        return True

    def play_round(self, player_move):
        if self.over:
            return self.state

        if player_move not in {"rock", "paper", "scissors"}:
            self._set_result("Invalid move detected. Round skipped.")
            call_robot_speech_api("I did not see a valid move.")
            self._update_state()
            self._check_game_over()
            return self.state

        computer_move = random.choice(["rock", "paper", "scissors"])
        result = self._determine_winner(player_move, computer_move)

        self.last_time = time.time()
        self.state.update({
            "player_move": player_move,
            "computer_move": computer_move,
            "result": result
        })
        self._update_state()

        self._check_game_over()
        return self.state

    def _determine_winner(self, player, computer):
        call_robot_gesture_api(computer)
        time.sleep(1) # Give robot time to start gesture before announcing result

        if player == computer:
            call_robot_speech_api("It's a draw!")
            return "Draw!"
        win_conditions = {
            ("rock", "scissors"),
            ("paper", "rock"),
            ("scissors", "paper")
        }
        if (player, computer) in win_conditions:
            self.score["player"] += 1
            call_robot_speech_api("You win this round!")
            return "You Win!"
        else:
            self.score["computer"] += 1
            call_robot_speech_api("I win this round!")
            return "Computer Wins!"
    
    def _check_game_over(self, force_end=False):
        player_wins = self.score["player"]
        computer_wins = self.score["computer"]
        majority = self.total_rounds // 2 + 1

        if force_end or self.round >= self.total_rounds or player_wins >= majority or computer_wins >= majority:
            self.over = True
            if player_wins > computer_wins:
                winner_msg = "Congratulations, you won the game!"
            elif computer_wins > player_wins:
                winner_msg = "I win the game! Better luck next time."
            else:
                winner_msg = "The game is a tie!"
            
            self.state["result"] = f"Game Over! {winner_msg}"
            call_robot_speech_api(winner_msg)
            print(f"[GAME OVER] {self.state['result']}")
            self._update_state()


# === Interface ===
game_manager = GameManager()
def prepare_round():
    return game_manager.start_round()

def play_round(player_move: str):
    return game_manager.play_round(player_move)

def reset_game():
    game_manager.reset()