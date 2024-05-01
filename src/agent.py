import random as rando
from main import *
import time
import numpy as np
import threading
from random import randrange as rand
import pygame, sys
import random
import keras
keras.layers.Dense(32)

import tensorflow as tf
global batch_size
batch_size = 64
def run_tetris_app(tetris_app):
    tetris_app.run()

def build_model(num_actions, input_shape=(308,)):
    model = keras.Sequential([
        keras.layers.Dense(32, activation='relu', input_shape=input_shape),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(num_actions)
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss=keras.losses.MeanSquaredError())

    return model

class tetris_agent(keras.Model):
    def __init__(self, num_actions):
        super(tetris_agent, self).__init__()
        self.model = build_model(num_actions, input_shape=(308,))

    def call(self, inputs):
        # Ensure the input shape matches the model's input shape
        if len(inputs.shape) == 2:
            inputs = tf.expand_dims(inputs, axis=0)
        if len(inputs.shape) == 3:
            inputs = tf.squeeze(inputs, axis=0)
            # Add batch dimension if missing
        return self.model(inputs)

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



    def reward_function(self, tetris_app):
        # # Get the user's score and the number of lines cleared
        # score = tetris_app.score
        lines_cleared = tetris_app.lines
        #
        # # Calculate bumpiness
        # bumpiness_reward = self.bumpiness(tetris_app)
        #
        # # Calculate compactness
        # compactness_reward = self.compactness(tetris_app)
        #
        # # Calculate holiness
        # holiness = self.num_holes(tetris_app)
        # # Calculate the total reward as a weighted sum of all components
        # total_reward = lines_cleared*100 - bumpiness_reward - self.sum_max_column_heights(tetris_app) - holiness + tetris_app.num_pieces + self.flatness(tetris_app)
        #


        return  lines_cleared*10

    def train(self, discount_factor=.99):
        epsilon = 0.1
        optimizer = keras.optimizers.Adam(learning_rate=0.005)
        loss_fn = keras.losses.MeanSquaredError()
        replay_buffer = []
        tetris_app = TetrisApp()
        tetris_thread = threading.Thread(target=run_tetris_app, args=(tetris_app,))
        tetris_thread.start()  # Start the Tetris game loop in a separate thread
        time.sleep(1)
        counter = 0
        # Agent interacts with the TetrisApp
        while True:
            counter += 1


            print(counter)
            # Perform agent actions
            state = self.encode_state(tetris_app)  # Get the current state
            padded_state = self.pad_input(state, target_shape=(22, 14))
            padded_state = np.array(padded_state)  # Convert state to NumPy array
            padded_state = np.expand_dims(padded_state, axis=0)  # Add batch dimension

            action = self.get_action(padded_state, epsilon)

            print(action)
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

print(tf.keras.layers.Dense(32, activation='relu'))
# Example usage:
agent = tetris_agent(num_actions=5)
agent.train()
