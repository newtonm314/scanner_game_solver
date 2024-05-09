import numpy as np
from collections import deque
import random
import tensorflow as tf
from tensorflow.keras.layers import Dense


class ScannerAgent:
    def __init__(
        self,
        env,
        learning_rate: float,
        initial_exploration_rate: float,
        minimum_exploration_rate: float,
        sync_rate: int,
        discount_factor: float = 0.97,
        batch_size: int = int(64),
    ):

        self.env = env
        self.lr = learning_rate
        self.initial_exploration_rate = initial_exploration_rate
        self.minimum_exploration_rate = minimum_exploration_rate
        self.sync_rate = sync_rate
        self.discount_factor = discount_factor
        self.batch_size = batch_size

        self.exploration_rate = self.initial_exploration_rate

        self.memory = deque(maxlen=512)

        # observation_space_shape = self.env.observation_space.shape
        # action_space_n = self.env.action_space.n
        observation_space_shape = (27,)
        action_space_n = 4

        self.policy_model = ScannerAgent.DQN(observation_space_shape, action_space_n)
        self.policy_model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr))
        self.target_model = ScannerAgent.DQN(observation_space_shape, action_space_n)
        self.target_model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr))

        self.sync_models()

    def DQN(observation_space_shape, action_space_n):
        return tf.keras.Sequential([
            Dense(24, input_shape=observation_space_shape, activation='relu'),
            # Dense(12, activation='relu'),
            Dense(action_space_n, activation='linear')
        ])

    # class DQN(tf.keras.Model):
    #     def __init__(self, observation_shape, action_n):
    #         super().__init__()
    #         self.observation_shape = observation_shape
    #         self.action_n = action_n

    #         self.dense1 = Dense(24, input_shape=observation_shape, activation='relu')
    #         self.dense2 = Dense(12, activation='relu')
    #         self.dense3 = Dense(action_n, activation='linear')

    #     def call(self, inputs):
    #         x = self.dense1(inputs)
    #         x = self.dense2(x)
    #         return self.dense3(x)
        
    #     def get_config(self):
    #         return {"observation_shape": self.observation_shape, "action_n": self.action_n}
        
    def sync_models(self):
        self.target_model.set_weights(self.policy_model.get_weights()) 
        # print("Syncing models")

    def get_action(self, obs) -> int:
        if np.random.random() < self.exploration_rate:
            return self.env.action_space.sample()
        else:
            q_values = self.policy_model.predict(obs,verbose=0)
            return np.argmax(q_values, axis=1)
        

    def store_sars(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        predictions = np.array(self.policy_model.predict(np.array(states),verbose=0),dtype='float64')

        targets = rewards + (not dones)*(self.discount_factor * np.amax(self.target_model.predict(np.array(next_states),verbose=0)))
        
        targets_vec = predictions
        for i in range(self.batch_size):
            targets_vec[i,actions[i]] = targets[i]

        self.policy_model.fit(np.array(states), np.array(targets_vec), epochs=1, verbose=0)

        batch_error = np.mean((targets - np.argmax(predictions,axis=1))**2)
        return batch_error

    def decay_epsilon(self, epsilon_decay):
        self.exploration_rate = min(self.initial_exploration_rate, max(self.minimum_exploration_rate, self.exploration_rate - epsilon_decay))

if __name__ == '__main__':
    try:
        import gymnasium as gym
        from tensorflow.keras.models import load_model
        import scanner_game

        env = gym.make('scanner_game/ScanWorld-v0', render_mode="human", field_size=5, scan_radius=1)
        env = gym.wrappers.FlattenObservation(env)

        agent = ScannerAgent(
            env=env,
            learning_rate=0.01,
            initial_exploration_rate=1,
            minimum_exploration_rate=0.1,
        )        
        # agent.model = load_model('model.keras')
        agent.policy_model = load_model('policy_model.keras')
        agent.target_model = load_model('target_model.keras')
        while True:
            obs, info = env.reset()
            done_episode = False
            while not done_episode:
                action = np.reshape(agent.get_action(np.reshape(obs,(1,-1))),(1,))
                next_obs, reward, terminated, truncated, info = env.step(action[0])
                obs = next_obs
                # time.sleep(0.1)
                done_episode = terminated or truncated
    except KeyboardInterrupt:
        pass