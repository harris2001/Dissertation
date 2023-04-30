import numpy as np
import matplotlib.pyplot as plt
import torch

import set_path

set_path.append_sys_path()

import argparse

import rela
import hanalearn
import r2d2

LAMBDA = 2 # Poisson Distribution lambda (changes percentage of agents playing each policy)
LEVELS = 5 # Cognitive hierarchies levels
EPSILON = 0.1
ALPHA = 0.1
BETA = 7
ENVIRONMENTS = 80
THREADS = 10
SEED = 123456
MAX_TRACE = 80 # There could be no more than 80 actions per game in Hanabi

# Use this for cognitive hierarchies
def assign_policy():
    level = np.random.choice(5,p=np.random.poisson())
    return level

# Use this function to generate epsilon 
def episode_exploration():
    episodes = []
    for i in range(ENVIRONMENTS):
        epsilon = ALPHA ** (1 +  BETA * i/(ENVIRONMENTS-1))
        if epsilon < 1e-6:
            epsilon = 0
        episodes.append(epsilon)
    return episodes

def parse_args():
    parser = argparse.ArgumentParser(description="train dqn on hanabi")
    parser.add_argument("--save_dir", type=str, default="exps/exp1")
    parser.add_argument("--method", type=str, default="vdn")
    parser.add_argument("--shuffle_color", type=int, default=0)
    parser.add_argument("--aux_weight", type=float, default=0)
    parser.add_argument("--boltzmann_act", type=int, default=0)
    parser.add_argument("--min_t", type=float, default=1e-3)
    parser.add_argument("--max_t", type=float, default=1e-1)
    parser.add_argument("--num_t", type=int, default=80)
    parser.add_argument("--hide_action", type=int, default=0)
    parser.add_argument("--off_belief", type=int, default=0)
    parser.add_argument("--belief_model", type=str, default="None")
    parser.add_argument("--num_fict_sample", type=int, default=10)
    parser.add_argument("--belief_device", type=str, default="cuda:1")

    parser.add_argument("--load_model", type=str, default="")
    parser.add_argument("--clone_bot", type=str, default="", help="behavior clone loss")
    parser.add_argument("--clone_weight", type=float, default=0.0)
    parser.add_argument("--clone_t", type=float, default=0.02)

    parser.add_argument("--seed", type=int, default=10001)
    parser.add_argument("--gamma", type=float, default=0.999, help="discount factor")
    parser.add_argument(
        "--eta", type=float, default=0.9, help="eta for aggregate priority"
    )
    parser.add_argument("--train_bomb", type=int, default=0)
    parser.add_argument("--eval_bomb", type=int, default=0)
    parser.add_argument("--sad", type=int, default=0)
    parser.add_argument("--num_player", type=int, default=2)

    # optimization/training settings
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--eps", type=float, default=1.5e-5, help="Adam epsilon")
    parser.add_argument("--grad_clip", type=float, default=5, help="max grad norm")
    parser.add_argument("--num_lstm_layer", type=int, default=2)
    parser.add_argument("--rnn_hid_dim", type=int, default=512)
    parser.add_argument(
        "--net", type=str, default="publ-lstm", help="publ-lstm/ffwd/lstm"
    )

    parser.add_argument("--train_device", type=str, default="cuda:0")
    parser.add_argument("--batchsize", type=int, default=128)
    parser.add_argument("--num_epoch", type=int, default=5000)
    parser.add_argument("--epoch_len", type=int, default=1000)
    parser.add_argument("--num_update_between_sync", type=int, default=2500)

    # DQN settings
    parser.add_argument("--multi_step", type=int, default=3)

    # replay buffer settings
    parser.add_argument("--burn_in_frames", type=int, default=10000)
    parser.add_argument("--replay_buffer_size", type=int, default=100000)
    parser.add_argument(
        "--priority_exponent", type=float, default=0.9, help="alpha in p-replay"
    )
    parser.add_argument(
        "--priority_weight", type=float, default=0.6, help="beta in p-replay"
    )
    parser.add_argument("--max_len", type=int, default=80, help="max seq len")
    parser.add_argument("--prefetch", type=int, default=3, help="#prefetch batch")

    # thread setting
    parser.add_argument("--num_thread", type=int, default=10, help="#thread_loop")
    parser.add_argument("--num_game_per_thread", type=int, default=40)

    # actor setting
    parser.add_argument("--act_base_eps", type=float, default=0.1)
    parser.add_argument("--act_eps_alpha", type=float, default=7)
    parser.add_argument("--act_device", type=str, default="cuda:1")
    parser.add_argument("--actor_sync_freq", type=int, default=10)

    args = parser.parse_args()
    if args.off_belief == 1:
        args.method = "iql"
        args.multi_step = 1
        assert args.net in ["publ-lstm"], "should only use publ-lstm style network"
        assert not args.shuffle_color
    assert args.method in ["vdn", "iql"]
    return args

# Main entry point
if __name__=="__main__":
    torch.backends.cudnn.benchmark = True
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    explore_eps = episode_exploration()
    expected_eps = np.mean(explore_eps)

    # Create games
    games = []
    for i in range(THREADS*ENVIRONMENTS):
        params = {
            "players": "5",
            "seed": str(SEED + i),
            "bomb": "0",
            "hand_size": "5",
            "random_start_player": "0",
        }
        game = hanalearn.HanabiEnv(
            params,
            MAX_TRACE,
            False,
        )
        games.append(game)
    
    #Create agent
    agent = r2d2.R2D2Agent(
        True, #allow value-decomposition networks
        args.multi_step,
        args.gamma,
        args.eta,
        args.train_device,
        games[0].feature_size(args.sad),
        args.rnn_hid_dim,
        games[0].num_action(),
        args.net,
        args.num_lstm_layer,
        args.boltzmann_act,
        False,  # uniform priority
        args.off_belief,
    )
    # updates target net from online net
    agent.sync_target_with_online()
    agent = agent.to(device)
    optim = torch.optim.Adam(agent.online_net.parameters(), lr=args.lr, eps=args.eps)
    # print(agent)
    eval_agent = agent.clone(device, {"vdn": False, "boltzmann_act": False})
