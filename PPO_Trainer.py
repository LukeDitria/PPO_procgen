from procgen import ProcgenEnv
import os
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

import Helpers.Helper_Functions as hf

class PpoTrainer():
    def __init__(self, args, actor_critic):
        self.args = args

        # Set the Pytorch device - GPU highly recommended
        self.device = torch.device(self.args.gpu if torch.cuda.is_available() else "cpu")

        # Calculate the mini batch size to use during training, Procgen baselines use a huge batch size - 2048
        # to speed up training, you can use a smaller one if your GPU can't handle it
        self.mini_batch_size = (self.args.num_steps * self.args.num_envs) // self.args.num_mini_batch

        # Initialize frame/update step counters and loss logger dictionary
        self.frame_indx = 0
        self.training_updates = 0
        self.loss_logger = {"critic": [], "actor": [], "actor_entropy": []}

        # Create a file save name and save path
        self.exp_name = self.args.env_name + "_PPO_" + self.args.save_name + str(self.args.num_levels) + "_" +\
                        self.args.dist_mode + "_200M_" + str(self.args.exp_indx)
        self.save_path = os.path.join(self.args.save_dir + "/Models", self.exp_name + ".pt")

        # Create the data buffer
        # designed so that you can "relatively" easily add new data types if required during training - needs work
        self.ppo_data_to_log = ("states", "values", "actions", "log_probs", "returns", "advantages", "rewards")
        self.data_buffer = hf.ReplayBuffer(self.ppo_data_to_log, int(self.args.num_steps * self.args.num_envs),
                                                self.mini_batch_size, self.device)

        # Initialize model and optimizer
        self.ppo_net = actor_critic(self.args.state_history*3, self.args.num_acts, self.args.width).to(self.device)
        self.ppo_optimizer = optim.Adam(self.ppo_net.parameters(), lr=self.args.lr)

        # Load checkpoint if load_checkpoint fag is set
        if self.args.load_checkpoint:
            self.load_checkpoint()
        else:
            # If checkpoint does exist and load_checkpoint = False
            # Raise an error to prevent accidental overwriting
            if os.path.isfile(self.save_path):
                raise ValueError("Warning Checkpoint exists")
            else:
                print("Starting from scratch")

    def load_checkpoint(self):
        if os.path.isfile(self.save_path):
            # load checkpoint
            check_point = torch.load(self.save_path)

            self.ppo_net.load_state_dict(check_point['agent_state_dict'])
            self.ppo_optimizer.load_state_dict(check_point['agent_optimizer_state_dict'])
            self.frame_indx = check_point['frames']
            self.training_updates = check_point['updates']
            self.loss_logger = check_point['losses']

            print("Checkpoint loaded, starting from epoch:", self.frame_indx)
        else:
            # Raise Error if it does not exist
            raise ValueError("Checkpoint Does not exist")

    def save_checkpoint(self):
        if not os.path.isdir(self.args.save_dir + "/Models"):
            os.mkdir(self.args.save_dir + "/Models")

        # Checkpoint is saved as a python dictionary
        torch.save({
            'frames': self.frame_indx,
            'updates': self.training_updates,
            'agent_state_dict': self.ppo_net.state_dict(),
            'agent_optimizer_state_dict': self.ppo_optimizer.state_dict(),
            'losses': self.loss_logger,
            'args': self.args.__dict__
        }, self.save_path)

    def ppo_update(self, clip_param=0.2):
        for data_batch in self.data_buffer:
            # PPO update!
            new_dist, new_value = self.ppo_net(data_batch["states"])  # Forward pass of input state observation

            # Determine expectation over the batch of the action distribution entropy for entropy bonus
            # Entropy bonus - increases the "entropy" of the action distribution, aka the "randomness"
            # This ensures that the actor does not converge to taking the same action and
            # maintains some ability for "exploration" of the policy
            entropy = new_dist.entropy().mean()

            ########### Policy Gradient update for actor with clipping - PPO #############
            # Work out the probability (log probability) that the agent will NOW take
            # the action it took during the rollout
            # We assume there has been some optimisation steps between when the action was taken and now so the
            # probability has probably changed
            new_log_probs = new_dist.log_prob(data_batch["actions"])

            # Calculate the ratio of new/old action probability (remember we have log probabilities here)
            # log(new_prob) - log(old_prob) = log(new_prob/old_prob)
            # exp(log(new_prob/old_prob)) = new_prob/old_prob
            ratio = (new_log_probs - data_batch["log_probs"]).exp()

            # We want to MAXIMISE the (Advantage X Ratio)
            # If the advantage is positive this corresponds to INCREASING the probability of taking that action
            # If the advantage is negative this corresponds to DECREASING the probability of taking that action
            surr1 = ratio * data_batch["advantages"]

            # We use the ratio of new/old action probabilities (not just the log probability of the action like in
            # vanilla policy gradients) so that if there is a large difference between the probabilities then we can
            # take a larger/smaller update step
            # EG: If we want to decrease the probability of taking an action but the new action probability
            # is now higher than it was before we can take a larger update step to correct this
            #
            # PPO goes a bit further, as if we simply update update the Advantage X Ratio we will sometimes
            # get very large or very small policy updates when we don't want them
            # EG1: If we want to increase the probability of taking an action but the new action probability
            # is now higher than it was before we will take a larger step, however if the action probability is
            # already higher we don't need to keep increasing it
            #
            # EG2: You can also consider the opposite case where we want to decrease the action probability
            # but the probability has already decreased, in this case we will take a smaller step than before,
            # which is also not desirable
            #
            # PPO therefore clips the upper bound of the ratio when the advantage is positive
            # and clips the lower bound of the ratio when the advantage is negative so our steps are not too large
            # or too small when necessary
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * data_batch["advantages"]
            actor_loss = torch.min(surr1, surr2).mean()

            ########### Value Function update for critic with clipping #############
            # We can do a similar thing for the value function update, if the new value function estimate is close to
            # the returns we don't have to do as big an update
            vf_loss1 = (new_value - data_batch["returns"]).pow(2.)
            vpredclipped = data_batch["values"] + torch.clamp(new_value - data_batch["values"], -clip_param, clip_param)
            vf_loss2 = (vpredclipped - data_batch["returns"]).pow(2.)
            critic_loss = torch.max(vf_loss1, vf_loss2).mean()

            # These techniques allow us to do multiple epochs of our data without huge update steps throwing off our
            # policy/value function (gradient explosion etc).
            # It can also help prevent "over-fitting" to a single batch of observations etc, 
            # RL boot-straps itself and the noisy "ground truth" targets (if you can call them that) will
            # shift overtime and we need to make sure our actor-critic can quickly adapt, over-fitting to a single batch
            # of observations will prevent that
            agent_loss = critic_loss - actor_loss - self.args.ent_loss * entropy

            self.ppo_optimizer.zero_grad()
            agent_loss.backward()
            nn.utils.clip_grad_norm_(self.ppo_net.parameters(), 40)  # Clip grad to further prevent large updates
            self.ppo_optimizer.step()

            # Log the losses
            self.loss_logger["critic"].append(critic_loss.item())
            self.loss_logger["actor"].append(actor_loss.item())
            self.loss_logger["actor_entropy"].append(entropy.item())

    def segment_rollout(self, states, actions, rewards, values, returns, log_probs):
        # Turn deque to list and concat the tensors along the batch dimension

        new_data = {"states": torch.cat(list(states)),
                    "actions": torch.cat(list(actions)),
                    "rewards": torch.cat(list(rewards)),
                    "returns": torch.cat(list(returns)),
                    "values": torch.cat(list(values)),
                    "log_probs": torch.cat(list(log_probs))
                    }
        
        # Add to the data buffer
        self.data_buffer_transfer(new_data)

    def data_buffer_transfer(self, new_data):
        self.frame_indx += new_data["states"].shape[0]  # Update the number of frames seen

        # Add each data type to the data buffer
        for keys, data in new_data.items():
            self.data_buffer.data_log(keys, data)

        advantage = (new_data["returns"] - new_data["values"]).squeeze(1)  # Calculate the advantage
        # Normalise the advantage (for training stability) and add to buffer
        self.data_buffer.data_log("advantages", (advantage - advantage.mean()) / (advantage.std() + 1e-8))

    def collect_rollouts(self):
        print("Rollout Start")
        # re-initialise the environments so we start from new games, not from where we left off last time
        # For Procgen "reset" does not reset the envs to the start of a new games

        envs = ProcgenEnv(num_envs=self.args.num_envs, env_name=self.args.env_name, start_level=0,
                          num_levels=self.args.num_levels, distribution_mode=self.args.dist_mode)

        # Initialise state
        start_state = envs.reset()
        state = hf.state_to_tensor(start_state, self.device)

        # Create data loggers - deques a bit faster than lists?
        log_probs = deque()
        values = deque()
        states = deque()
        actions = deque()
        rewards = deque()
        masks = deque()
        step = 0
        done = np.zeros(self.args.num_envs)

        with torch.no_grad():  # Don't need computational graph for roll-outs
            while step < self.args.num_steps:
                #  Masks so we can separate out multiple games in the same environment
                current_mask = torch.FloatTensor(1 - done).unsqueeze(1).to(self.device)
                masks.append(current_mask)

                dist, value = self.ppo_net(state)  # Forward pass of actor-critic model
                action = dist.sample()  # Sample action from distribution

                # Take the next step in the env
                next_state, reward, done, env_info = envs.step(action.cpu().numpy())

                # Work out reward, we are just setting the reward to be either
                # -1, 1 or 0 for negative, positive or no reward
                ppo_reward = torch.FloatTensor(((reward > 0).astype("float64") - (reward < 0).astype("float64")))

                # Log data
                log_prob = dist.log_prob(action)
                log_probs.append(log_prob)
                states.append(state)
                actions.append(action)
                values.append(value)
                rewards.append(ppo_reward.unsqueeze(1).to(self.device))

                state = hf.state_to_tensor(next_state, self.device)
                step += 1

            # Get value at time step T+1
            _, next_value = self.ppo_net(state)
            # Calculate the returns/gae
            returns = hf.compute_gae(next_value, rewards, masks, values,
                                     gamma=self.args.ppo_gamma, tau=self.args.ppo_tau)

            # organise and add data to data buffer
            self.segment_rollout(states, actions, rewards, values, returns, log_probs)

    def train_model(self, lock):
        start_of_time = time.time()

        while self.frame_indx < self.args.max_frames:
            # Linear decay of learning rate
            hf.lr_linear(self.args.max_frames, self.frame_indx, self.args.lr, self.ppo_optimizer)

            self.ppo_net.eval()  # Set to eval mode - doesn't do anything for Impala CNN
            self.collect_rollouts()  # Perform "roll-outs"

            print("Training")
            self.ppo_net.train()  # Set to train mode - doesn't do anything for Impala CNN

            for _ in range(self.args.ppo_epochs):  # Perform training epochs
                self.ppo_update()
            self.training_updates += 1  # Update training steps counter

            # print time to completion estimate
            time_to_end = ((time.time() - start_of_time) / self.frame_indx) * (self.args.max_frames - self.frame_indx)
            print("Frames: [%d/%d]" % (self.frame_indx, self.args.max_frames))
            print("Time to end: %.2f" % (time_to_end / 3600))

            # Save checkpoint
            with lock:
                print("Saving")
                self.save_checkpoint()
