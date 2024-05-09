import time
import scanner_game
import gymnasium as gym
from scanner_agent import ScannerAgent
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import os
from math import ceil


class ScannerTrainer():
    def __init__(self, num_envs=1, render=None):
        self.num_envs = num_envs

        learning_rate = 0.01
        start_epsilon = 1.0
        final_epsilon = 0.1

        self.env = gym.vector.AsyncVectorEnv([
            lambda: 
                gym.wrappers.FlattenObservation(
                gym.make('scanner_game/ScanWorld-v0', 
                         render_mode=render, 
                         field_size=5, 
                         scan_radius=1, 
                         max_episode_steps=100))
        ]*num_envs)

        self.agent = ScannerAgent(
            env=self.env,
            learning_rate=learning_rate,
            initial_exploration_rate=start_epsilon,
            minimum_exploration_rate=final_epsilon,
            sync_rate=num_envs*50
        )


    def train(self, n_episodes):
        epsilon_decay = self.agent.initial_exploration_rate / (n_episodes / 2.0)
        
        self.epsilon_hist = []
        self.episode_reward_hist = []
        self.episode_lengths_hist = []
        self.train_error_hist = []

        sync_rate = self.num_envs*50

        episode_count = 0
        step_count = 0
        current_episode_lengths = np.zeros(self.num_envs,)
        obs, info = self.env.reset()
        with tqdm(total=n_episodes) as pbar:
            while episode_count < n_episodes:
                action = self.agent.get_action(obs)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                current_episode_lengths += 1
                for i in range(self.num_envs):
                    if terminated[i] or truncated[i]:
                        self.agent.store_sars(obs[i], action[i], reward[i], info['final_observation'][i], terminated[i])
                        
                        self.episode_reward_hist.append(info['final_info'][i]['episode_reward'])
                        self.epsilon_hist.append(self.agent.exploration_rate)

                        if terminated[i]:
                            self.agent.decay_epsilon(epsilon_decay)
                        # if truncated[i]:
                        #     self.agent.decay_epsilon(-2*epsilon_decay)

                        self.episode_lengths_hist.append(current_episode_lengths[i])
                        current_episode_lengths[i] = 0
                        episode_count += 1
                        pbar.update(1)
                    else:
                        self.agent.store_sars(obs[i], action[i], reward[i], next_obs[i], terminated[i])
                    
                err = self.agent.learn()
                if err is not None:
                    self.train_error_hist.append(err)

                step_count += self.num_envs
                if (step_count % sync_rate) == 0:
                    self.agent.sync_models()

                obs = next_obs

        print("Completed in {} steps".format(step_count))
        self.env.close()
        self.agent.policy_model.save('policy_model.keras', save_format='tf')
        self.agent.target_model.save('target_model.keras', save_format='tf')

    def results(self, rolling_length=1):
        fig, axs = plt.subplots(ncols=4, figsize=(12, 5))
        axs[0].set_title("Episode reward")
        reward_moving_average = (
            np.convolve(
                np.array(self.episode_reward_hist).flatten(), np.ones(rolling_length), mode="same"
            )
            / rolling_length
        )
        axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
        axs[1].set_title("Episode lengths")
        length_moving_average = (
            np.convolve(
                np.array(self.episode_lengths_hist).flatten(), np.ones(rolling_length), mode="same"
            )
            / rolling_length
        )
        axs[1].plot(range(len(length_moving_average)), length_moving_average)
        axs[2].set_title("Training Error")
        training_error_moving_average = (
            np.convolve(np.array(self.train_error_hist), np.ones(rolling_length), mode="same")
            / rolling_length
        )
        axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
        plt.tight_layout()
        plt.show()
        axs[3].set_title("Epsilon")
        axs[3].plot(range(len(self.epsilon_hist)), self.epsilon_hist)


if __name__ == '__main__':
    trainer = ScannerTrainer(num_envs=100)
    trainer.train(n_episodes=int(4e3))
    trainer.results(rolling_length=int(1e2))

    


    

