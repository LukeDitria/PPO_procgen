import gym
import os
import numpy as np
import pandas as pd
import torch
import time

import Helpers.Helper_Functions as hf


class PpoTester():
    def __init__(self, args, actor_critic):
        self.args = args

        self.device = torch.device(self.args.gpu if torch.cuda.is_available() else "cpu")
        self.ppo_net = actor_critic(self.args.state_history*3, self.args.num_acts, self.args.width).to(self.device)

        # Create a file save name and save path
        self.exp_name = self.args.env_name + "_PPO_" + self.args.save_name + str(self.args.num_levels) + "_" + \
                        self.args.dist_mode + "_" + str(int(self.args.max_frames // 1e6)) + "M_" + str(
            self.args.exp_indx)

        self.save_path = os.path.join(self.args.save_dir + "/Models", self.exp_name + ".pt")
        self.save_data_path = os.path.join(self.args.save_dir + "/Data", self.exp_name)

        self.frame_idx = 0
        self.training_updates = 0

        self.d = {"Frames": [], "Updates": [], "Training": [], "Testing": []}
        self.score_logger = pd.DataFrame(self.d)

        if not os.path.isdir(self.args.save_dir + "/Data"):
            os.mkdir(self.args.save_dir + "/Data")

        if os.path.isfile(self.save_data_path + '_scores.csv'):
            self.score_logger = pd.read_csv(self.save_data_path + '_scores.csv')
            print("Data loaded from save")

    def load_checkpoint(self):
        if os.path.isfile(self.save_path):
            while True:
                try:
                    check_point = torch.load(self.save_path)
                    self.ppo_net.load_state_dict(check_point['agent_state_dict'])
                    self.frame_idx = check_point['frames']
                    self.training_updates = check_point['updates']
                    print("Models Loaded")
                    break
                except:
                    time.sleep(0.01)
                    continue
        else:
            print("Model not Found")

    def test_env(self, env):
        start_state = env.reset()
        state = hf.state_to_tensor(start_state, self.device)

        done = False
        total_reward = 0
        steps = 0
        with torch.no_grad():
            while not done:
                dist, _ = self.ppo_net(state)
                action = dist.sample().item()

                next_state, reward, done, _ = env.step(action)
                total_reward += reward

                state = hf.state_to_tensor(next_state, self.device)
                steps += 1

        return total_reward

    def test_model(self, lock):
        print("Testing Models")
        while self.frame_idx < self.args.max_frames:
            with lock:
                self.load_checkpoint()

            if self.frame_idx > 0:
                print("Frames Seen %d" % self.frame_idx)
                test_rewards = []
                train_rewards = []

                print("Testing Train environments")
                for i in range(min(self.args.num_levels, 500)):
                    env = gym.make("procgen:procgen-" + self.args.env_name + "-v0",
                                   start_level=i, num_levels=1, distribution_mode=self.args.dist_mode)
                    train_rewards.append(self.test_env(env))
                    time.sleep(0.05)

                print("Testing Test environments")
                for i in range(min(self.args.num_levels, 500)):
                    env = gym.make("procgen:procgen-" + self.args.env_name + "-v0",
                                   start_level=self.args.num_levels + i, num_levels=1,
                                   distribution_mode=self.args.dist_mode)

                    test_rewards.append(self.test_env(env))
                    time.sleep(0.05)

                print("Training/Testing [%.2f/%.2f]" % (np.mean(train_rewards), np.mean(test_rewards)))

                d = {"Frames": [self.frame_idx], "Updates": [self.training_updates],
                     "Training": [np.mean(train_rewards)],
                     "Testing": [np.mean(test_rewards)]}

                score = pd.DataFrame(d)
                self.score_logger = self.score_logger.append(score)

                self.score_logger.to_csv(self.save_data_path + '_scores.csv', index=False)
                print("Data saved")
            else:
                time.sleep(5)
