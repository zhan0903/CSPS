from utils.memory import ray_get_and_free,ReplayBuffer
import gym,ray
import numpy as np
import time,os
import torch
import random
import statistics
import engines
from utils.logx import EpochLogger


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Walker2d-v2')
    parser.add_argument('--hid', type=int, default=300)
    parser.add_argument('--l', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.99)
    # parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--version', type=str, default=0)
    parser.add_argument('--max_timesteps', type=int, default=1e6)
    parser.add_argument('--sac_init_training', type=int, default=5e4)
    parser.add_argument('--sac_init_sample', type=int, default=1e4)
    parser.add_argument('--gap', type=int, default=100) # transfer gap

    parser.add_argument('--start_steps', type=int, default=1e4) # 
    # parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--max_ep_len', type=int, default=1000)
    parser.add_argument('--replay_size', type=int, default=1000000)
    parser.add_argument('--optimize_alpha', type=bool, default=False)
    parser.add_argument('--alpha',type=float,default=0.2)
    parser.add_argument('--iteration_steps',type=int,default=5e3)
    # parser.add_argument('--count_iteration',type=int,default=5000)
    parser.add_argument('--local_size',type=int,default=20000)
    parser.add_argument('--discount', type=float, default=0.2)

    # CEM
    parser.add_argument('--pop_size', type=int, default=10)
    parser.add_argument('--sigma_init', default=1e-3, type=float)
    parser.add_argument('--damp', default=1e-3, type=float)
    parser.add_argument('--damp_limit', default=1e-5, type=float)
    parser.add_argument('--elitism', dest="elitism",  action='store_true') # defult False

    parser.add_argument('--seed', '-s', type=int, default=0)
    args = parser.parse_args()

    ray.init(ignore_reinit_error=True, object_store_memory=2000000000)

    env = gym.make(args.env)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    from utils.logx import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.version,args.env,args.seed)
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(vars(args))

    max_timesteps = int(args.max_timesteps)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    global_mem = ReplayBuffer(obs_dim=state_dim,act_dim=action_dim,size=args.replay_size)
    local_mem = ReplayBuffer(obs_dim=state_dim,act_dim=action_dim,size=args.local_size)

    engine = engines.EngineCSPC(args,global_mem,local_mem,logger)

    engine.train(max_timesteps)
    ray.shutdown()
    



    