import random
import argparse
# from Networks.CNN_nets import SplitActorCritic as actor_critic

from Networks.IMPALA_CNN import ImpalaCnn64 as actor_critic
from Visualise_jupyter import GetRepresentationVisualisation
import copy

parser = argparse.ArgumentParser(description="Train PPO model.")

#string args
parser.add_argument("--dist_mode", help="difficulty", default="hard")
parser.add_argument("--num_levels", help="Number of train/test levels", type=int,  default=500)
parser.add_argument("--save_dir", help="Number game environments for training", default=".")

parser.add_argument("--save_name", help="experiment info", default="PPO_")
parser.add_argument("--gpu", help ="gpu indx", type=int,  default=0)
parser.add_argument("--env_name", help="Environment Name", default="coinrun")

#int args
parser.add_argument("--max_frames", help="number of training frames", type=int, default=200000000)
parser.add_argument("--state_size", help ="state representation size", type=int, default=512)
parser.add_argument("--test_lvls", help ="Number of train/test levels", type=int, default=20)
parser.add_argument("--exp_indx", help="Experiment Index", type=int, default=0)
parser.add_argument("--width", help="channel width", type=int, default=16)
parser.add_argument("--num_acts", help="Number of discrete actions", type=int, default=15)
parser.add_argument("--ppo_epochs", help="num training epochs", type=int, default=3)

parser.add_argument("--attention", action='store_true', help="is it an attention model?")
parser.add_argument("--ppo_gamma", help="ppo gae gamma", type=float, default=0.999)

args = parser.parse_args()

if __name__ == '__main__':

    visualiser = GetRepresentationVisualisation(args, actor_critic)
    visualiser.get_data(10000000, args.test_lvls, True)
    # visualiser.gaussian_mixture()
    # visualiser.terminal_state_figure()
    visualiser.plot_values()

    # visualiser.plot_rewards()
    # #
    visualiser.get_tsne()
    visualiser.create_plots()
    # visualiser.record_rollout()

