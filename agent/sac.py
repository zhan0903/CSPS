# Adjusted from https://github.com/openai/spinningup
from agent import _core
import gym,torch
import numpy as np
import random
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SAC(object):
    def __init__(self,
        args,
        actor_critic=_core.ActorCritic_SAC,
        ac_kwargs=dict(),
        steps_per_epoch=5000,
        epochs=200, 
        replay_size=int(1e6),
        gamma=0.99,
        polyak=0.995,
        lr=1e-3,
        alpha=0.2,
        optimize_alpha=True,
        batch_size=100,
        start_steps=10000,
        max_ep_len=1000,
        logger_kwargs=dict(),
        save_freq=1):

        self.env = gym.make(args.env)
        self.env.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]
        ac_kwargs['action_space'] = self.env.action_space
        
        self.total_timesteps = 0
        self.optimize_alpha = optimize_alpha
        self.args = args

        self.device = device
        self.learned_step = 0
        self.which_one = 0
        self.gamma = gamma
        self.alpha = alpha
        self.fitnesses = []
        self.ep_lens = []
        self.value = None
        self.visit_count = 0
        self.polyak = polyak
        self.main = actor_critic(in_features=obs_dim, **ac_kwargs)
        self.target = actor_critic(in_features=obs_dim, **ac_kwargs)
        self.explore_steps = 0

        self.pi_optimizer = torch.optim.Adam(self.main.policy.parameters(), lr=lr)
        self.value_params = list(self.main.vf_mlp.parameters()) + list(
        self.main.q1.parameters()) + list(self.main.q2.parameters())
        self.value_optimizer = torch.optim.Adam(self.value_params, lr=lr)

        # alpha optimimer
        if optimize_alpha:
            self.target_entropy = -np.prod(self.env.action_space.shape).item()
            self.log_alpha = torch.zeros(1, requires_grad=True,device="cuda")
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

        # Initializing targets to match main variables
        self.target.vf_mlp.load_state_dict(self.main.vf_mlp.state_dict())

    def get_actor_params(self):
        self.main.policy.cpu()
        return self.main.policy.net.state_dict(),self.main.policy.mu.state_dict()

    def get_value_params(self):
        self.main.vf_mlp.cpu()
        return self.main.vf_mlp.state_dict()

    def update(self, steps, global_mem, local_mem, batch_size=100,gamma=0.99,soft_tau=1e-2):
        self.main.cuda()
        self.target.cuda()

        for it in range(steps):
            p = random.random()
            if p <= 0.3 and local_mem.size > int(self.args.sac_init_sample):
                replay_mem = local_mem
            else:
                replay_mem = global_mem

            batch = replay_mem.sample_batch(batch_size)

            (obs1, obs2, acts, rews, done) = (torch.tensor(batch['obs1']),
                                            torch.tensor(batch['obs2']),
                                            torch.tensor(batch['acts']),
                                            torch.tensor(batch['rews']),
                                            torch.tensor(batch['done']))

            obs1 = torch.FloatTensor(obs1).to(device)
            obs2 = torch.FloatTensor(obs2).to(device)
            acts = torch.FloatTensor(acts).to(device)
            rews = torch.FloatTensor(rews).to(device)
            done = torch.FloatTensor(done).to(device)

            _, _, logp_pi, q1, q2, q1_pi, q2_pi, v = self.main(obs1, acts)
                
            v_targ = self.target.vf_mlp(obs2)

            # Min Double-Q:
            min_q_pi = torch.min(q1_pi, q2_pi)

            # Targets for Q and V regression
            q_backup = (rews + gamma * (1 - done) * v_targ).detach()
            v_backup = (min_q_pi - self.alpha * logp_pi).detach()

            # Soft actor-critic losses
            pi_loss = (self.alpha * logp_pi - min_q_pi).mean()
            q1_loss = 0.5 * F.mse_loss(q1, q_backup)
            q2_loss = 0.5 * F.mse_loss(q2, q_backup)
            v_loss = 0.5 * F.mse_loss(v, v_backup)
            value_loss = q1_loss + q2_loss + v_loss

            # Policy train op
            self.pi_optimizer.zero_grad()
            pi_loss.backward()
            self.pi_optimizer.step()

            # Value train op
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

            # Polyak averaging for target parameters
            for p_main, p_target in zip(self.main.vf_mlp.parameters(),
                                        self.target.vf_mlp.parameters()):
                p_target.data.copy_(self.polyak * p_target.data +
                                    (1 - self.polyak) * p_main.data)

    def _explore(self,return_gap,global_mem,local_mem,max_ep_len=1000):
        self.main.policy.cpu()
        episode_reward = 0
        episode_timesteps = 0
        done = False
        one_episode = []
        obs = self.env.reset()

        while True:  
            if return_gap == -np.inf and self.sac_timesteps < self.args.start_steps:
                action = self.env.action_space.sample()
            else:
                action = self.main.policy.select_action(np.array(obs),_eval=False)

            new_obs, reward, done, _ = self.env.step(action)

            episode_timesteps += 1
            self.sac_timesteps += 1
            episode_reward += reward
            done = False if episode_timesteps == max_ep_len else done
            one_episode.append((obs, action, reward, new_obs, done))

            if done or (episode_timesteps == max_ep_len): 
                global_mem.add_epsodes([one_episode])

                if episode_reward > return_gap and return_gap != -np.inf:
                    local_mem.add_epsodes([one_episode])
                break

            obs = new_obs

        return episode_timesteps

    def train(self,iteration,global_mem,local_mem,return_gap):
        self.sac_timesteps = 0
        while self.sac_timesteps < iteration:
            steps = self._explore(return_gap,global_mem,local_mem)
            self.update(steps,global_mem,local_mem)
        return self.sac_timesteps

    def test_policy(self, eval_episodes=5):
        eval_env = gym.make(self.args.env)
        eval_env.seed(self.args.seed + 100)
        avg_reward = 0.
        self.main.policy.cpu()
        
        for _ in range(eval_episodes):
                state, done = eval_env.reset(), False
                while not done:
                        action = self.main.policy.select_action(np.array(state))
                        state, reward, done, _ = eval_env.step(action)
                        avg_reward += reward

        avg_reward /= eval_episodes

        return avg_reward


