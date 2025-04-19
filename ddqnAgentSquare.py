"""
Name: Priyanshu Ranka (NUID: 002305396)
Project: DDQN for Car Racing
Course: CS 5180 - Reinforcement Learning
Professor: Prof. Robert Platt
Semester: Spring 2025
Description: This code implements a Double Deep Q-Network (DDQN) agent for the Car Racing environment using Keras.
"""

# Import necessary libraries
from keras.layers import Dense, Activation
from keras.models import Sequential, load_model
from keras.optimizers import Adam
import numpy as np
import tensorflow as tf

# Class to store the replay buffer
# This class is used to store the experiences of the agent in the environment
class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions, discrete=False):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.discrete = discrete
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        dtype = np.int8 if self.discrete else np.float32
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=dtype)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    # This function is used to store the experiences in the replay buffer
    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        # store one hot encoding of actions, if appropriate
        if self.discrete:
            actions = np.zeros(self.action_memory.shape[1])
            actions[action] = 1.0
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action
        
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - done
        self.mem_cntr += 1

    # This function is used to sample a batch of experiences from the replay buffer
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal

# This class implements the Double Deep Q-Network (DDQN) agent
# The DDQN agent uses two neural networks to estimate the Q-values for the actions
class DDQNAgent(object):
    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size,
                 input_dims, epsilon_dec=0.995,  epsilon_end=0.10,
                 mem_size=25000, fname="C:\\Users\\yashr\\Desktop\\DDQN-Car-Racing\\ddqn_model.h5", replace_target=25):
        self.action_space = [i for i in range(n_actions)]
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec  
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname
        self.replace_target = replace_target
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions, discrete=True)
       
        self.brain_eval = Brain(input_dims, n_actions, batch_size)
        self.brain_target = Brain(input_dims, n_actions, batch_size)

    # This function is used to store the experiences in the replay buffer
    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    # This function is used to choose an action based on the current state
    def choose_action(self, state):

        state = np.array(state)
        state = state[np.newaxis, :]

        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            actions = self.brain_eval.predict(state)
            action = np.argmax(actions)

        return action

    # This function is used to learn from the experiences stored in the replay buffer
    def learn(self):
        if self.memory.mem_cntr > self.batch_size:
            state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

            action_values = np.array(self.action_space, dtype=np.int8)
            action_indices = np.dot(action, action_values)

            q_next = self.brain_target.predict(new_state)
            q_eval = self.brain_eval.predict(new_state)
            q_pred = self.brain_eval.predict(state)

            max_actions = np.argmax(q_eval, axis=1)

            q_target = q_pred

            batch_index = np.arange(self.batch_size, dtype=np.int32)

            q_target[batch_index, action_indices] = reward + self.gamma*q_next[batch_index, max_actions.astype(int)]*done

            _ = self.brain_eval.train(state, q_target)

            self.epsilon = self.epsilon*self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min

    # This function is used to update the target network
    def update_network_parameters(self):
        self.brain_target.copy_weights(self.brain_eval)

    # This function is used to save the model in the new .keras format
    def save_model(self):
        # Save the model in the new .keras format
        self.brain_eval.model.save('ddqn_model.keras')
        self.brain_target.model.save('ddqn_model_target.keras')  # If you also want to save the target network
        print("Model saved successfully as ddqn_model.keras")

    # This function is used to load the model from the .keras format
    def load_model(self):
        try:
            # Load the model from the .keras format
            self.brain_eval.model = tf.keras.models.load_model('ddqn_model.keras')
            self.brain_target.model = tf.keras.models.load_model('ddqn_model_target.keras')  # If you also want to load the target model
            print("Model loaded successfully from ddqn_model.keras")
        except Exception as e:
            print(f"Error loading model: {e}")

# This class implements the neural network used by the DDQN agent
# The neural network is a simple feedforward neural network with one hidden layer
class Brain:
    def __init__(self, NbrStates, NbrActions, batch_size = 256):
        self.NbrStates = NbrStates
        self.NbrActions = NbrActions
        self.batch_size = batch_size
        self.model = self.createModel()
    
    # This function is used to create the neural network model
    def createModel(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu)) # 256 neurons in the hidden layer 
        model.add(tf.keras.layers.Dense(self.NbrActions))        #, activation=tf.nn.softmax))  # no activation
        model.compile(loss = "mse", optimizer="adam")

        return model
    
    # This function is used to train the model
    def train(self, x, y, epoch = 1, verbose = 0):
        self.model.fit(x, y, batch_size = self.batch_size , verbose = verbose)

    # This function is used to predict the Q-values for a given state
    def predict(self, s):
        return self.model.predict(s)

    # This function is used to predict the Q-values for a given state
    def predictOne(self, s):
        return self.model.predict(tf.reshape(s, [1, self.NbrStates])).flatten()
    
    # This function is used to copy the weights from one model to another
    def copy_weights(self, TrainNet):
        variables1 = self.model.trainable_variables
        variables2 = TrainNet.model.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())