
import pygame
import numpy as np
from GameEnv import RacingEnv
from ddqn_keras_new import DDQNAgent  # or import from dqn_agent if needed

# Choose the model to run
USE_DDQN = True  # Set False to use DQN

# Initialize the agent based on your choice
if USE_DDQN:
   ddqn_agent = DDQNAgent(
        alpha=0.001, gamma=0.99, n_actions=5, epsilon=0.0, batch_size=128,
        input_dims=19, epsilon_dec=0.995, epsilon_end=0.10, mem_size=25000,
        fname="ddqn_model_track.h5"  # Load the saved model
    )
   ddqn_agent.load_model()  # Load the pre-trained model

# Initialize the environment
env = RacingEnv()
env.fps = 60  # Set a visible frame rate (30 or 60)

def play_episode():
    env.reset()  # Reset the environment for the episode
    done = False
    state, _, _ = env.step(0)  # Start the environment with a no-op action (0)
    state = np.array(state)
    total_reward = 0

    while not done:
        # Process events (to handle quitting the simulation)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # Choose the next action using the agent (using the model learned so far)
        action = ddqn_agent.choose_action(state)

        # Perform the action in the environment
        new_state, reward, done = env.step(action)

        # Update the state for the next iteration
        state = np.array(new_state) if new_state is not None else state
        total_reward += reward

        # Render the current state and action of the environment
        env.render(action)

    # Print the result of the episode
    print(f"Episode finished with total reward: {total_reward:.2f}")

if __name__ == "__main__":
    NUM_EPISODES = 3  # Specify how many episodes you want to visualize

    for _ in range(NUM_EPISODES):
        play_episode()  # Play each episode

    env.close()  # Close the environment after playing all episodes
