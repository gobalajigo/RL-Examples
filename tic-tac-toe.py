import numpy as np
import random

class TicTacToe:
    def __init__(self):
        self.board = [' '] * 9  # 3x3 board as a list
        self.current_player = 'X'  # X always starts

    def reset(self):
        self.board = [' '] * 9
        self.current_player = 'X'
        return self.get_state()

    def get_state(self):
        return tuple(self.board)  # Board state as a tuple (hashable)

    def available_actions(self):
        return [i for i in range(9) if self.board[i] == ' ']

    def make_move(self, action):
        if self.board[action] == ' ':
            self.board[action] = self.current_player
            self.current_player = 'O' if self.current_player == 'X' else 'X'
            return True
        return False

    def is_winner(self, player):
        win_conditions = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]             # Diagonals
        ]
        return any(all(self.board[i] == player for i in condition) for condition in win_conditions)

    def is_draw(self):
        return ' ' not in self.board and not self.is_winner('X') and not self.is_winner('O')

    def get_reward(self, player):
        if self.is_winner(player):
            return 1
        elif self.is_winner('O' if player == 'X' else 'X'):
            return -1
        elif self.is_draw():
            return 0
        return 0

    def game_over(self):
        return self.is_winner('X') or self.is_winner('O') or self.is_draw()

# Reinforcement Learning Agent
class RLAgent:
    def __init__(self, player, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.player = player
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.q_table = {}  # State-action value table

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def update_q_value(self, state, action, reward, next_state):
        max_next_q = max([self.get_q_value(next_state, a) for a in range(9)], default=0.0)
        current_q = self.get_q_value(state, action)
        self.q_table[(state, action)] = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)

    def choose_action(self, state, available_actions):
        if random.random() < self.epsilon:  # Explore
            return random.choice(available_actions)
        # Exploit
        q_values = [self.get_q_value(state, a) for a in available_actions]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(available_actions, q_values) if q == max_q]
        return random.choice(best_actions)

# Training the agent
def train_agent(episodes=10000):
    game = TicTacToe()
    agent = RLAgent('X')

    for episode in range(episodes):
        state = game.reset()
        while not game.game_over():
            available_actions = game.available_actions()
            action = agent.choose_action(state, available_actions)
            game.make_move(action)

            reward = game.get_reward(agent.player)
            next_state = game.get_state()

            agent.update_q_value(state, action, reward, next_state)
            state = next_state

            if game.game_over():
                break
    return agent

# Playing a game against the trained agent
def play_game(agent):
    game = TicTacToe()
    state = game.reset()
    print("Starting a game of Tic Tac Toe!")

    while not game.game_over():
        print("\nCurrent Board:")
        print_board(game.board)

        if game.current_player == agent.player:
            action = agent.choose_action(state, game.available_actions())
            print(f"Agent chooses position {action}")
        else:
            action = int(input("Your move (0-8): "))

        if game.make_move(action):
            state = game.get_state()
        else:
            print("Invalid move. Try again.")

    print("\nFinal Board:")
    print_board(game.board)
    if game.is_winner('X'):
        print("X wins!")
    elif game.is_winner('O'):
        print("O wins!")
    else:
        print("It's a draw!")

def print_board(board):
    for i in range(3):
        print(board[i * 3:(i + 1) * 3])

# Train and play
trained_agent = train_agent(episodes=10000)
play_game(trained_agent)
