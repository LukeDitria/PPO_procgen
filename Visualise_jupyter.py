import gym
import os
import numpy as np
import cv2
import math
import copy
import random
import torch
import torchvision
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch.nn.functional as F
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
import Helpers.Helper_Functions as hf

class GetRepresentationVisualisation():
    def __init__(self, args, actor_critic):
        self.args = args
        self.device = torch.device(self.args.gpu if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(self.args.gpu)

        self.ppo_net = actor_critic(3, self.args.num_acts, self.args.width).to(self.device)
        self.ppo_net.eval()
        self.exp_name = self.args.env_name + "_PPO_" + self.args.save_name + str(self.args.num_levels) + "_" +\
                        self.args.dist_mode + "_" + str(int(self.args.max_frames//1e6)) + "M_" + str(self.args.exp_indx)
        self.save_path = os.path.join(self.args.save_dir + "/Models", self.exp_name + ".pt")
        self.save_data_path = os.path.join(self.args.save_dir + "/Data", self.exp_name)

        self.level_number_logger = []
        self.states_logger = []
        self.state_representations_logger = []
        self.actions_logger = []
        self.rewards_logger = []
        self.values_logger = []
        self.train_test_logger = []
        self.tsne_embedding = []
        self.terminal_states = []
        self.number_of_steps = []
        self.masks_logger = []

        self.frame_indx = 0

        self.load_checkpoint()

        print(self.ppo_net.mean_offset)

    def load_checkpoint(self):
        print(self.save_path)
        if os.path.isfile(self.save_path):
            # load checkpoint
            check_point = torch.load(self.save_path)
            self.ppo_net.load_state_dict(check_point['agent_state_dict'])
            self.frame_indx = check_point['frames']
            # self.mean_error = check_point['mean_error']

            print("Checkpoint loaded, starting from epoch:", self.frame_indx)
        else:
            # Raise Error if it does not exist
            raise ValueError("Checkpoint Does not exist")

    def reset_data(self):
        self.level_number_logger = []
        self.states_logger = []
        self.state_representations_logger = []
        self.actions_logger = []
        self.rewards_logger = []
        self.values_logger = []
        self.train_test_logger = []
        self.tsne_embedding = []
        self.terminal_states = []
        self.number_of_steps = []
        self.masks_logger = []

    def get_data(self, start_test_level, max_levels=10, test_levels=True):
        print("Collecting data")
        for i in range(min(max_levels, self.args.num_levels)):
            env = gym.make("procgen:procgen-" + self.args.env_name + "-v0", start_level=i, num_levels=1,
                           distribution_mode=self.args.dist_mode)
            self.collect_rollout(env, i)
            print("Level %d" % i)

        if test_levels:
            for i in range(min(max_levels, self.args.num_levels)):
                env = gym.make("procgen:procgen-" + self.args.env_name + "-v0",
                               start_level=start_test_level + i,
                               num_levels=1,
                               distribution_mode=self.args.dist_mode)
                self.collect_rollout(env, i+min(max_levels, self.args.num_levels), 1)
                print("Level %d" % i)

        self.state_representations_logger = torch.cat(self.state_representations_logger).numpy()
        self.states_logger = torch.cat(self.states_logger)
        self.terminal_states = torch.cat(self.terminal_states)

    def collect_rollout(self, env, level_num, train_test=0, random_acts=False):
        start_state = env.reset()
        # print(env.unwrapped.env.env.combos)
        state = hf.state_to_tensor(start_state, self.device)
        self.ppo_net.eval()

        step = 0
        done = False
        with torch.no_grad():
            while not done:

                dist, value = self.ppo_net(state)  # Forward pass of actor-critic model
                # action = dist.sample().argmax(1)
                action = dist.sample()  # Sample action from distribution
                next_state, reward, done, env_info = env.step(action.item())

                # Work out reward, we are just setting the reward to be either
                # -1, 1 or 0 for negative, positive or no reward
                ppo_reward = torch.FloatTensor([((reward > 0).astype("float64") - (reward < 0).astype("float64"))])
                ppo_reward = ppo_reward.unsqueeze(1)

                self.level_number_logger.append(level_num)
                self.states_logger.append(state.cpu())
                self.state_representations_logger.append(self.ppo_net.state.cpu().reshape(1, -1))
                self.actions_logger.append(action.cpu())
                self.rewards_logger.append(ppo_reward)
                self.values_logger.append(value.cpu())
                self.train_test_logger.append(train_test)

                state = hf.state_to_tensor(next_state, self.device)
                current_mask = torch.FloatTensor([1 - done]).unsqueeze(1)
                self.masks_logger.append(current_mask)

                step += 1
            self.terminal_states.append(state.cpu())
            self.number_of_steps.append(step)

    def get_tsne(self):
        print("Performing TNSE")
        if len(self.state_representations_logger) > 0:
            if self.state_representations_logger.shape[1] > 2:
                if self.state_representations_logger.shape[1] > 50:
                    pca = PCA(n_components=50)
                    pca.fit(self.state_representations_logger)
                    reps_50d = pca.transform(self.state_representations_logger)
                    self.tsne_embedding = TSNE(n_components=2).fit_transform(reps_50d)
                else:
                    self.tsne_embedding = TSNE(n_components=2).fit_transform(self.state_representations_logger)
            else:
                self.tsne_embedding = self.state_representations_logger
        else:
            print("No data!")
            return None

    def onclick(self, event):
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              ('double' if event.dblclick else 'single', event.button,
               event.x, event.y, event.xdata, event.ydata))

        point = np.expand_dims(np.array([event.xdata, event.ydata]), axis = 0)
        point_indexs = self.find_nearest(point)
        self.get_state_figure(point_indexs)
        if self.args.attention:
            self.get_attention_figure(point_indexs)

    def gaussian_mixture(self, centers=40):
        gaussian_mixture = GaussianMixture(n_components=centers)
        cluster_labels = gaussian_mixture.fit_predict(self.state_representations_logger)
        cluster_gen = 0
        for i in range(centers):
            sum_train = np.sum(np.array(self.train_test_logger)[cluster_labels == i])
            cluster_gen += np.abs((sum_train / np.sum(cluster_labels == i)) - 0.5) * 2
        print("Cluster Generalization Ratio:", cluster_gen/centers)

        # print(cluster_labels.shape)
        # plt.scatter(self.tsne_embedding[:, 0], self.tsne_embedding[:, 1], c=cluster_labels, cmap="tab20")
        # print(cluster_labels)
        # plt.show()

    def create_plots(self):
        print("Plotting")
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True)
        fig.subplots_adjust(hspace=0, wspace=0)
        ax1.scatter(self.tsne_embedding[:, 0], self.tsne_embedding[:, 1], c=self.level_number_logger, cmap="tab20")
        ax1.set_title("Level Distribution")
        ax1.set_xticks([])
        ax1.set_yticks([])
        ###############################################################################################################
        ax2.scatter(self.tsne_embedding[:, 0], self.tsne_embedding[:, 1], c=self.actions_logger, cmap="tab20")
        ax2.set_title("Action Distribution")
        ax2.set_xticks([])
        ax2.set_yticks([])
        ###############################################################################################################
        scatter = ax3.scatter(self.tsne_embedding[:, 0], self.tsne_embedding[:, 1], c=torch.cat(self.values_logger).numpy())
        ax3.set_title("Value Distribution", y=-0.1)
        cbaxes = inset_axes(ax3, width="30%", height="3%", loc=2)
        fig.colorbar(scatter, cax=cbaxes, ax=ax3, orientation="horizontal", shrink=0.3)
        ax3.set_xticks([])
        ax3.set_yticks([])
        ###############################################################################################################
        train_levels = np.array(self.train_test_logger) == 0
        ax4.scatter(self.tsne_embedding[:, 0][train_levels], self.tsne_embedding[:, 1][train_levels],
                              label="Train")

        test_levels = np.array(self.train_test_logger) == 1
        ax4.scatter(self.tsne_embedding[:, 0][test_levels], self.tsne_embedding[:, 1][test_levels],
                              label="Test")
        ax4.set_title("Test-Train Level Distribution", y=-0.1)
        ax4.legend(loc='upper left', numpoints=1, ncol=3, fontsize=8)
        ax4.set_xticks([])
        ax4.set_yticks([])
        ###############################################################################################################
        fig.set_size_inches(18.5, 10.5)

        cid = fig.canvas.mpl_connect('button_press_event', self.onclick)
        plt.show()
        fig.savefig("Plots/" + self.exp_name + '_plot.png')

    def find_nearest(self, x_y, n=25):
        if len(self.tsne_embedding) > 0:
            closest_point = np.argsort(np.sum(np.abs((self.tsne_embedding - x_y)), 1))[0]
            closest_representation = self.state_representations_logger[closest_point, :]
            closest_points = np.argsort(np.sum(np.abs((self.state_representations_logger - closest_representation)), 1))[:n]
            return closest_points
        else:
            print("No Data!")
            return None

    def get_state_figure(self, state_indices):
        top_n_states = self.states_logger[state_indices, :3]
        image_grid = torchvision.utils.make_grid(top_n_states, 5)
        image_grid = image_grid.numpy().transpose((1, 2, 0))
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(image_grid)
        plt.show()
        plt.savefig("Top_" + str(len(state_indices)) +".png")

    def terminal_state_figure(self):
        image_grid = torchvision.utils.make_grid(self.terminal_states, 5)
        image_grid = image_grid.numpy().transpose((1, 2, 0))
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(image_grid)
        plt.show()
        plt.savefig("Terminal_states.png")

    def plot_values(self):
        # returns = hf.compute_gae(next_value, rewards, masks, values)

        returns = hf.compute_returns(self.rewards_logger, self.masks_logger, gamma=self.args.ppo_gamma)
        returns = torch.cat(list(returns)).numpy()

        fig = plt.figure(figsize=(20, 10))
        plt.plot(torch.cat(self.values_logger).numpy())
        plt.plot(returns)

        plt.plot(1 - np.array(torch.cat(self.masks_logger)), c="k", linewidth=4)

        plt.legend(["Value", "Basic Returns"])
        plt.savefig("Plots/" + self.exp_name + "_Values.png")
        plt.show()

    def plot_rewards(self):
        fig = plt.figure(figsize=(20, 10))

        plt.plot(self.rewards_logger)
        plt.plot(self.masks_logger, linestyle="-")
        plt.savefig("Plots/Rewards.png")
        plt.show()

    def record_rollout(self):
        self.reset_data()
        
        env = gym.make("procgen:procgen-" + self.args.env_name + "-v0", start_level=500,
                       num_levels=1, distribution_mode=self.args.dist_mode)
        self.collect_rollout(env, 20, random_acts=False)

        height = 64
        width = 64
        if self.args.attention:
            UpNN = torch.nn.UpsamplingBilinear2d(scale_factor=8)
            height *= 2

        Video_name = "Videos/test_video.avi"
        video = cv2.VideoWriter(Video_name, cv2.VideoWriter_fourcc(*"MJPG"), 15, (width, height))
        print(len(self.states_logger))
        with torch.no_grad():
            for i in range(len(self.states_logger)):
                img = self.states_logger[i][0, :3].cpu().numpy()
                if self.args.attention:
                    atten_mask = UpNN(self.attention_maps_logger[i])[0].cpu().numpy()
                    masked_img = img * atten_mask
                    img = np.concatenate((img, masked_img), 1)

                image = (img.transpose((1, 2, 0)) * 255).astype(np.uint8)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                video.write(image)
            video.release()
