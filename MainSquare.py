"""
Name: Priyanshu Ranka (NUID: 002305396)
Project: DDQN for Car Racing
Course: CS 5180 - Reinforcement Learning
Professor: Prof. Robert Platt
Semester: Spring 2025
Description: This file contains the main function to train the DDQN agent on the car racing environment of Race Track.

"""

# Importing the necessary libraries
import GameEnv1
import pygame
import numpy as np
from ddqnAgentSquare import DDQNAgent
from collections import deque
import random, math

# Variables for the game environment and agent
TOTAL_GAMETIME = 500            # Max game time for one episode
N_EPISODES = 1000               #10000
REPLACE_TARGET = 25             #used 50 earlier

# Game environment initialization
game = GameEnv1.RacingEnv()
game.fps = 60
GameTime = 0 
GameHistory = []
renderFlag = False

# Agent initialization
ddqn_agent = DDQNAgent(alpha=0.001, gamma=0.99, n_actions=5, epsilon=1.00, epsilon_end=0.1, 
                       epsilon_dec=0.995, replace_target= REPLACE_TARGET, batch_size=128, input_dims=19)
ddqn_scores = []
eps_history = []

# Function to run the training loop
def run():
    for e in range(N_EPISODES):
        # Resetting the game environment and agent for each episode
        game.reset() #reset env 
        done = False
        score = 0
        counter = 0
        
        # Initialize the observation and action
        obs, reward, done = game.step(0)
        observation = np.array(obs)

        gtime = 0               # Set game time back to 0
        renderFlag = False      # Render flag to control rendering

        # Render the game every 20 episodes
        if e % 20 == 0 and e > 0: 
            renderFlag = True

        # Quit when done
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: 
                    return

            # Choose action using the DDQN agent
            action = ddqn_agent.choose_action(observation)
            obs, reward, done = game.step(action)
            obs = np.array(obs)

            # This is a countdown if no reward is collected the car will be done within 100 ticks
            if reward == 0:
                counter += 1
                if counter > 100:
                    done = True
            else:
                counter = 0

            score += reward

            # Store the game history
            ddqn_agent.remember(observation, action, reward, obs, int(done))
            observation = obs
            ddqn_agent.learn()
            
            gtime += 1

            if gtime >= TOTAL_GAMETIME:
                done = True

            if renderFlag:
                game.render(action)
                
        # Store the game history
        eps_history.append(ddqn_agent.epsilon)
        ddqn_scores.append(score)
        avg_score = np.mean(ddqn_scores[max(0, e-100):(e+1)])

        if e % REPLACE_TARGET == 0 and e > REPLACE_TARGET:
            ddqn_agent.update_network_parameters()

        # Save the model every 10 episodes
        if e % 10 == 0 and e > 10:
            ddqn_agent.save_model()
            print("save model")
            
            print('episode: ', e,'score: %.2f' % score,
                ' average score %.2f' % avg_score,
                ' epsolon: ', ddqn_agent.epsilon,
                ' memory size', ddqn_agent.memory.mem_cntr % ddqn_agent.memory.mem_size)   

# Main function to run the training loop 
if __name__ == '__main__':
    run()        