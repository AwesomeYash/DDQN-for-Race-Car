
import pygame
import numpy as np
import tensorflow as tf
from ddqn_keras import DDQNAgent  # Assuming you have this from the previous code
from GameEnv1 import RacingEnv  # Assuming you have this from the previous code

def play_trained_model():
    # Initialize the environment
    game = RacingEnv()
    game.fps = 60
    game.reset()  # Reset the environment to the starting state

    # Load the trained model
    ddqn_agent = DDQNAgent(
        alpha=0.001, gamma=0.99, n_actions=5, epsilon=0.0, batch_size=128,
        input_dims=19, epsilon_dec=0.995, epsilon_end=0.10, mem_size=25000,
        fname="ddqn_model.keras"  # Load the saved model
    )
    ddqn_agent.load_model()  # Load the pre-trained model

    done = False
    state, _, _ = game.step(0)
    state = np.array(state)

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # Use the trained agent to choose an action
        action = ddqn_agent.choose_action(state)
        
        # Take the action in the environment
        new_state, reward, done = game.step(action)
        
        # Update the state for the next iteration
        state = np.array(new_state) if new_state is not None else state
        
        # Render the environment with the agent's action
        game.render(action)

        # You can optionally print the current episode score here if needed
        print(f"Action: {action}, Reward: {reward}, Episode Done: {done}")

    print("Episode Finished")

if __name__ == "__main__":
    play_trained_model()
