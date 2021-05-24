import argparse
from Networks.IMPALA_CNN import ImpalaCnn64 as actor_critic

import multiprocessing as mp
from PPO_Trainer import PpoTrainer
from PPO_Tester import PpoTester
import copy

parser = argparse.ArgumentParser(description="Train PPO model.")

# string args
parser.add_argument("--env_name", help="Environment Name", default="coinrun")
parser.add_argument("--dist_mode", help="difficulty", default="hard")
parser.add_argument("--save_name", help="experiment info", default="PPO_")
parser.add_argument("--save_dir", help="Number game environments for training", default=".")

# int args
parser.add_argument("--ppo_epochs", help="num training epochs", type=int, default=3)
parser.add_argument("--num_steps", help="num rollout steps", type=int, default=256)
parser.add_argument("--num_mini_batch", help="mini batches per epoch", type=int, default=8)
parser.add_argument("--state_history", help="concat frames used", type=int, default=1)
parser.add_argument("--state_size", help="state representation size", type=int, default=512)
parser.add_argument("--max_frames", help="number of training frames", type=int, default=200000000)
parser.add_argument("--width", help="channel width", type=int, default=16)
parser.add_argument("--num_levels", help="Number of train/test levels", type=int, default=500)
parser.add_argument("--num_envs", help="Number of game environments for training", type=int, default=64)
parser.add_argument("--num_acts", help="Number of discrete actions", type=int, default=15)

parser.add_argument("--gpu", help="gpu indx", type=int, default=0)
parser.add_argument("--exp_indx", help="Experiment Index", type=int, default=0)

# float args
parser.add_argument("--lr", help="learning rate", type=float, default=5e-4)
parser.add_argument("--ent_loss", help="entropy loss scale", type=float, default=0.01)
parser.add_argument("--gamma", help="gae gamma", type=float, default=0.99)
parser.add_argument("--ppo_tau", help="ppo gae tau", type=float, default=0.95)
parser.add_argument("--ppo_gamma", help="ppo gae gamma", type=float, default=0.99)

# bool args
parser.add_argument("--load_checkpoint", action='store_true', help="Load checkpoint or start from scratch")

args = parser.parse_args()

if __name__ == '__main__':
    mp.set_start_method('spawn')
    ppo_trainer = PpoTrainer(args, actor_critic)
    ppo_tester = PpoTester(copy.deepcopy(args), actor_critic)
    m = mp.Manager()

    l = m.Lock()  # create a lock so both processes don't try to access the save file at the same time
    # spawn a separate process for testing
    test_p = mp.Process(target=ppo_tester.test_model, args=(l,))

    test_p.start()
    train_p = ppo_trainer.train_model(l)

    test_p.join()
