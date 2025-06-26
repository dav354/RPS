import os
import random
import time
import requests
import threading

# === Config ===
ROBOT_API_BASE_URL = os.environ.get("PEPPER_IP")
COOLDOWN = 3
TOTAL_ROUNDS = 3

# === Async HTTP Utility ===
def make_async_request(url, method='GET', payload=None, timeout=2):
    def _request():
        try:
            if method == 'GET':
                requests.get(url, timeout=timeout)
            elif method == 'POST':
                requests.post(url, json=payload, timeout=timeout)
        except requests.RequestException as e:
            print(f"[❌] Request failed: {e}")
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
            "result": "",
            "score": self.score.copy(),
            "round": self.round,
            "game_over": self.over
        }
        call_robot_speech_api("Game reset.")
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
            "last_played": self.last_time
        })

    def start_round(self):
        if self.over:
            self._set_result("Game over. Please reset.")
            return False

        if self._cooldown_remaining() > 0:
            self._set_result(f"Cooldown... wait {self._cooldown_remaining():.1f}s")
            return False

        if self.round >= self.total_rounds:
            self.over = True
            self._set_result("Game over.")
            return False

        self.round += 1
        self.last_time = time.time()
        self.state.update({
            "player_move": "none",
            "computer_move": "none",
            "result": f"Round {self.round}/{self.total_rounds}: Waiting for player..."
        })
        self._update_state()

        call_robot_speech_api(f"Round {self.round}. Show your move!")
        call_robot_gesture_api("swing")
        print(f"[ROUND] Round {self.round}/{self.total_rounds} started")
        return True

    def play_round(self, player_move):
        if self.over:
            self._set_result("Game over.")
            return self.state

        if self._cooldown_remaining() > 0:
            self._set_result(f"Still in cooldown... wait {self._cooldown_remaining():.1f}s")
            return self.state

        if player_move not in {"rock", "paper", "scissors"}:
            self._set_result("Invalid move. Choose rock, paper, or scissors.")
            call_robot_speech_api("Invalid move.")
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
        call_robot_speech_api(computer)
        call_robot_gesture_api(computer)

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
            call_robot_speech_api("You win!")
            return "You Win!"
        else:
            self.score["computer"] += 1
            call_robot_speech_api("I win!")
            return "Computer Wins!"

    def _check_game_over(self):
        player_wins = self.score["player"]
        computer_wins = self.score["computer"]
        majority = self.total_rounds // 2 + 1

        if self.round == self.total_rounds or player_wins >= majority or computer_wins >= majority:
            self.over = True
            if player_wins > computer_wins:
                winner = "Player"
                call_robot_speech_api("Congratulations! You won!")
            elif computer_wins > player_wins:
                winner = "Computer"
                call_robot_speech_api("I win the game!")
            else:
                winner = "It's a tie!"
                call_robot_speech_api("It's a tie!")
            self.state["result"] += f" 🎉 Game Over! {winner} wins."
            print(f"[GAME OVER] {winner} wins.")
            self._update_state()

# === Interface ===
game_manager = GameManager()

def prepare_round():
    return game_manager.start_round()

def play_round(player_move: str):
    return game_manager.play_round(player_move)

def reset_game():
    game_manager.reset()
