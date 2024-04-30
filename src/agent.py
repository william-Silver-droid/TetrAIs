import random as rando
from main import *
import time
import numpy as np
import threading
from random import randrange as rand
import pygame, sys
import random

import tensorflow as tf
global batch_size
batch_size = 64
def run_tetris_app(tetris_app):
    tetris_app.run()

class tetris_agent(tf.keras.Model):
    def __init__(self, num_actions):
        super(tetris_agent, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(num_actions)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.output_layer(x)

    def move_left(self):
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_LEFT}))

    def move_right(self):
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_RIGHT}))

    def move_down(self):
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_DOWN}))

    def rotate(self):
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_UP}))

    def drop(self):
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_RETURN}))

    def toggle_pause(self):
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_p}))

    def start_game(self):
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_SPACE}))

    def take_action(self, out):
        if out == 0:
            self.move_left()
        if out == 1:
            self.move_right()
        if out == 2:
            self.rotate()
        if out == 3:
            self.move_down()
        if out == 4:
            self.drop()

    def encode_state(self, tetris_game):
        board = tetris_game.board
        next_stone = tetris_game.next_stone
        current_stone = tetris_game.stone


        # Flatten the nested lists
        flattened_board = [item for sublist in board for item in sublist]
        flattened_next_stone = [item for sublist in next_stone for item in sublist]
        flattened_current_stone = [item for sublist in current_stone for item in sublist]

        # Concatenate all lists into one continuous list
        state = flattened_board + flattened_next_stone + flattened_current_stone

        return state

    def pad_input(self, state, target_shape):
        # Pad the flattened state to match the target shape
        padded_state = np.pad(state, (0, target_shape[0] * target_shape[1] - len(state)), mode='constant')

        return padded_state


    def get_action(self, state, epsilon):
        if random.random() < epsilon:

            return random.randint(0, 4)  # Assuming there are 5 possible actions (0 to 4)
        q_values = self.predict(state)  # Get Q-values for the current state
        action = np.argmax(q_values)
        if action == 4: action -= 1
        return action

    def bumpiness(self, tetris_app):
        """
        Calculate the sum of differences in height of adjacent columns in the Tetris board.

        Parameters:
            tetris_app: Tetris application object.

        Returns:
            int: Sum of differences in height of adjacent columns.
        """
        board = tetris_app.board
        heights = [0] * (len(board[0]) + 2)  # Initialize list to store heights of columns (including edges)
        total_bumpiness = 0

        # Calculate the height of each interior column
        for col in range(1, len(board[0]) + 1):
            for row in range(len(board)):
                if board[row][col - 1] == 1:
                    heights[col] = len(board) - row
                    break

        # Compute the sum of differences in height between adjacent columns
        for i in range(1, len(heights) - 1):
            total_bumpiness += abs(heights[i] - heights[i + 1])

        return total_bumpiness

    def compactness(self, tetris_app):
        board = tetris_app.board
        compactness_reward = 0

        # Check each tile on the board
        for row in range(len(board)):
            for col in range(len(board[0])):
                if board[row][col] != 0:  # If the tile is filled
                    # Check cardinal adjacent tiles
                    adjacent_tiles = [
                        (row, col - 1),  # Left
                        (row, col + 1)  # Right
                    ]

                    # Check if all adjacent tiles are either filled or covered by the edges
                    all_adjacent_filled = all(
                        0 <= r < len(board) and 0 <= c < len(board[0]) and board[r][c] == 1
                        for r, c in adjacent_tiles
                    )

                    if all_adjacent_filled:
                        compactness_reward += 1

        return compactness_reward

    def sum_max_column_heights(self, tetris_app):
        """
        Sum all the maximum heights of each column in the Tetris board.

        Parameters:
            tetris_app: Tetris application object.

        Returns:
            int: Sum of the maximum heights of each column.
        """
        board = tetris_app.board
        max_heights = []

        # Calculate the width and height of the board
        height = len(board)
        width = len(board[0])

        # Iterate over each column to find its maximum height
        for col in range(width):
            max_height = 0
            for row in range(height):
                if board[row][col] != 0:
                    max_height = height - row
                    break
            max_heights.append(max_height)

        # Return the sum of the maximum heights of all columns
        return sum(max_heights)

    def flatness(self, tetris_app):
        board = tetris_app.board
        flatness_score = 0
        width = len(board[0])

        # Iterate through each row
        for row in board:
            # Count the number of filled tiles in the row
            num_filled_tiles = sum(1 for cell in row if cell ==1 )
            flatness_score += num_filled_tiles**2
        flatness_score**=(1/2)

        return flatness_score

    def num_holes(self, tetris_app):
        board = tetris_app.board
        num_holes = 0
        width = len(board[0])
        height = len(board)

        # Iterate through each column
        for col in range(width):
            # Find the row index of the first filled cell in the column
            filled_row = -1
            for row in range(height):
                if board[row][col] != 0:
                    filled_row = row
                    break

            # If a filled cell is found in the column
            if filled_row != -1:
                # Start counting holes from the row below the filled cell
                for row in range(filled_row + 1, height):
                    if board[row][col] == 0:
                        num_holes += 1

        return num_holes

    def calculate_row_bonus(self, tetris_app):
        board = tetris_app.board
        bonus = 0
        for row_index, row in enumerate(board):
            filled_tiles = sum(1 for tile in row if tile == 1)  # Count filled tiles in the row
            if filled_tiles > 0:
                bonus += (4 - row_index) * filled_tiles
        return bonus

    def calculate_max_height(self, tetris_app):
        board = tetris_app.board
        max_height = 0
        for row in board:
            height = len(row) - row.count(0) + row.count(2)  # Count non-empty tiles in the row
            if height > max_height:
                max_height = height
        return max_height

    def reward_function(self, tetris_app):
        # Get the user's score and the number of lines cleared
        score = tetris_app.score
        lines_cleared = tetris_app.lines

        # Calculate bumpiness
        bumpiness_reward = self.bumpiness(tetris_app)

        # Calculate compactness
        compactness_reward = self.compactness(tetris_app)

        # Calculate holiness
        holiness = self.num_holes(tetris_app)
        # Calculate the total reward as a weighted sum of all components
        total_reward = lines_cleared*100 - bumpiness_reward - self.sum_max_column_heights(tetris_app) - holiness + tetris_app.num_pieces + self.flatness(tetris_app)


        return self.calculate_row_bonus(tetris_app) + lines_cleared*10

    def train(self, discount_factor=.99):
        epsilon = 0.1
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
        loss_fn = tf.keras.losses.MeanSquaredError()
        replay_buffer = []
        tetris_app = TetrisApp()
        tetris_thread = threading.Thread(target=run_tetris_app, args=(tetris_app,))
        tetris_thread.start()  # Start the Tetris game loop in a separate thread
        time.sleep(1)
        counter = 0
        # Agent interacts with the TetrisApp
        while True:
            counter += 1


            # Perform agent actions
            state = self.encode_state(tetris_app)  # Get the current state
            padded_state = self.pad_input(state, target_shape=(22, 14))
            padded_state = np.array(padded_state)  # Convert state to NumPy array
            padded_state = np.expand_dims(padded_state, axis=0)  # Add batch dimension

            action = self.get_action(padded_state, epsilon)


            self.take_action(action)  # Take action in the environment
            self.move_down()
            # Capture next state, reward, and done flag
            next_state = self.encode_state(tetris_app)
            next_padded_state = self.pad_input(next_state, target_shape=(22, 14))
            next_padded_state = np.array(next_padded_state)
            next_padded_state = np.expand_dims(next_padded_state, axis=0)
            reward = self.reward_function(tetris_app)
            done = tetris_app.gameover

            # Append transition to replay buffer
            replay_buffer.append((padded_state, action, reward, next_padded_state, done))
            if done:
                epsilon *= .999  # Choose action with highest Q-value
                reward -= 100
                # Restart the game
                time.sleep(1)
                self.start_game()
                time.sleep(1)
                # If replay buffer is sufficiently filled, perform training
                if len(replay_buffer) >= batch_size:
                    minibatch = random.sample(replay_buffer, batch_size)
                    states, actions, rewards, next_states, dones = zip(*minibatch)
                    states = np.concatenate(states)
                    next_states = np.concatenate(next_states)

                    target_q_values = self.predict(states)
                    next_q_values = self.predict(next_states)

                    for i in range(batch_size):
                        if dones[i]:
                            target_q_values[i][actions[i]] = rewards[i]
                        else:
                            target_q_values[i][actions[i]] = rewards[i] + discount_factor * np.max(next_q_values[i])

                    # Compute loss
                    with tf.GradientTape() as tape:
                        predicted_q_values = self.call(states)
                        loss = loss_fn(target_q_values, predicted_q_values)

                    # Update Q-network parameters
                    gradients = tape.gradient(loss, self.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, self.trainable_variables))


    # def train_final(self, batch_size=32):
    #     # Define optimizer and loss function
    #     optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    #     loss_fn = tf.keras.losses.MeanSquaredError()
    #
    #     # Replay buffer to store experiences
    #     replay_buffer = []
    #
    #     # Training loop
    #     while not tetris_app.gameover:
    #         # Perform agent actions
    #         state = self.encode_state(tetris_app)
    #         action = self.get_action(state)
    #
    #         # Take action and observe next state and reward
    #         reward, next_state, done = self.take_action(action)
    #
    #         # Store experience in replay buffer
    #         replay_buffer.append((state, action, reward, next_state, done))
    #
    #         # Sample a minibatch from the replay buffer
    #         if len(replay_buffer) >= batch_size:
    #             minibatch = random.sample(replay_buffer, batch_size)
    #
    #             # Compute target Q-values for minibatch
    #             for state, action, reward, next_state, done in minibatch:
    #                 if done:
    #                     target_q_value = reward
    #                 else:
    #                     target_q_value = reward + self.discount_factor * np.max(self.predict(next_state))
    #                 target_q_values = self.predict(state)
    #                 target_q_values[0][action] = target_q_value
    #
    #                 # Compute loss
    #                 with tf.GradientTape() as tape:
    #                     predicted_q_values = self.call(state)
    #                     loss = loss_fn(target_q_values, predicted_q_values)
    #
    #                 # Update Q-network parameters
    #                 gradients = tape.gradient(loss, self.trainable_variables)
    #                 optimizer.apply_gradients(zip(gradients, self.trainable_variables))


# Define TetrisApp class with the necessary methods


# Define your TetrisApp and TetrisAgent classes here

# Function to run the Tetris game loop in a separate thread


# Example usage:
agent = tetris_agent(num_actions=5)
agent.train()
