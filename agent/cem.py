# Adjusted from https://github.com/apourchot/CEM-RL
import ray,gym
import numpy as np
import random,torch
from agent import _core
from copy import deepcopy


class sepCEM:
    """
    Cross-entropy methods.
    """

    def __init__(self, num_params,
                 mu_init=None,
                 sigma_init=1e-3,
                 pop_size=256,
                 damp=1e-3,
                 damp_limit=1e-5,
                 parents=None,
                 elitism=False,
                 antithetic=False):

        # misc
        self.num_params = num_params

        # distribution parameters
        if mu_init is None:
            self.mu = np.zeros(self.num_params)
        else:
            self.mu = np.array(mu_init)
        self.sigma = sigma_init
        self.damp = damp
        self.damp_limit = damp_limit
        self.tau = 0.95
        self.cov = self.sigma * np.ones(self.num_params)

        # elite stuff
        self.elitism = elitism
        self.elite = np.sqrt(self.sigma) * np.random.rand(self.num_params)
        self.elite_score = None

        # sampling stuff
        self.pop_size = pop_size
        self.antithetic = antithetic

        if self.antithetic:
            assert (self.pop_size % 2 == 0), "Population size must be even"
        if parents is None or parents <= 0:
            self.parents = pop_size // 2
        else:
            self.parents = parents
        self.weights = np.array([np.log((self.parents + 1) / i)
                                 for i in range(1, self.parents + 1)])
        self.weights /= self.weights.sum()

    def ask(self, pop_size):
        """
        Returns a list of candidates parameters
        """
        if self.antithetic and not pop_size % 2:
            epsilon_half = np.random.randn(pop_size // 2, self.num_params)
            epsilon = np.concatenate([epsilon_half, - epsilon_half])

        else:
            epsilon = np.random.randn(pop_size, self.num_params)

        inds = self.mu + epsilon * np.sqrt(self.cov)
        if self.elitism:
            inds[-1] = self.elite

        return inds

    def tell(self, solutions, scores):
        """
        Updates the distribution
        """
        scores = np.array(scores)
        scores *= -1
        idx_sorted = np.argsort(scores)

        old_mu = self.mu
        self.damp = self.damp * self.tau + (1 - self.tau) * self.damp_limit
        self.mu = self.weights @ solutions[idx_sorted[:self.parents]]

        z = (solutions[idx_sorted[:self.parents]] - old_mu)
        self.cov = 1 / self.parents * self.weights @ (
            z * z) + self.damp * np.ones(self.num_params)

        self.elite = solutions[idx_sorted[0]]
        self.elite_score = scores[idx_sorted[0]]
        # print(self.cov)

    def get_distrib_params(self):
        """
        Returns the parameters of the distrubtion:
        the mean and sigma
        """
        return np.copy(self.mu), np.copy(self.cov)

@ray.remote
class CEM:
    def __init__(self,args):
        self.env = gym.make(args.env)
        self.env.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

        state_dim = self.env.observation_space.shape[0]
        action_space = self.env.action_space
        self.args = args
        self.all_fitness = []
        self.explore_steps = 0
        self.sac_params_net = None
        self.ppo_params_net = None
        assert args.elitism == False

        self.actor = _core.Deterministic_CEM(state_dim,(400, 300), torch.relu, torch.tanh, action_space)
        
        self.es = sepCEM(self.actor.get_size(), mu_init=self.actor.get_params(), sigma_init=args.sigma_init, damp=args.damp, damp_limit=args.damp_limit,
                pop_size=args.pop_size, antithetic=not args.pop_size % 2, parents=args.pop_size // 2, elitism=args.elitism)
        self.es_params= self.es.ask(self.args.pop_size)


    def transfer_sac(self,sac_params_net,sac_params_mu):
        self.sac_params_net = sac_params_net
        self.sac_params_mu = sac_params_mu

    def transfer_ppo(self,ppo_params_net,ppo_params_mu):
        self.ppo_params_net = ppo_params_net
        self.ppo_params_mu = ppo_params_mu

    def get_actor_params(self): 
        self.actor.set_params(self.es.elite)
        return self.actor.state_dict()

    def update(self,global_mem=None,explore_timesteps=None,global_mem_h=None):
        self.es.tell(self.es_params, self.all_fitness)
    
    def _explore(self, policy, max_ep_len=1000):
        episode_reward = 0
        episode_timesteps = 0
        done = False
        experiences = []
        obs = self.env.reset()

        while True:  
            action = policy.select_action(np.array(obs),_eval=False)
            new_obs, reward, done, _ = self.env.step(action)

            episode_timesteps += 1
            episode_reward += reward
            done = False if episode_timesteps == max_ep_len else done
            experiences.append((obs, action, reward, new_obs, done))
            obs = new_obs
            if done or (episode_timesteps == max_ep_len): break

        return episode_timesteps,experiences,episode_reward

    def train(self, iteration, return_gap, sac_cem, ppo_cem):
        timesteps = 0
        global_mem_temp = []
        local_mem_temp = []
        
        if sac_cem:
            self.actor.net.load_state_dict(self.sac_params_net)
            self.actor.mu.load_state_dict(self.sac_params_mu)
            self.es_params[0] = self.actor.get_params()

        if ppo_cem:
            self.actor.net.load_state_dict(self.ppo_params_net)
            self.actor.mu.load_state_dict(self.ppo_params_mu)
            self.es_params[1] = self.actor.get_params()

        while timesteps < iteration:
            all_fitness = []

            for params in self.es_params:
                self.actor.set_params(params)
                steps, one_epsode,f = self._explore(self.actor)
                timesteps += steps
                all_fitness.append(f)
                if sac_cem:
                    if f > return_gap:
                        local_mem_temp.append(one_epsode)
                global_mem_temp.append(one_epsode)

            self.es.tell(self.es_params, all_fitness)
            self.es_params = self.es.ask(self.args.pop_size)

        return timesteps,global_mem_temp,local_mem_temp

    def test_policy(self, eval_episodes=5):
        self.actor.set_params(self.es.elite)
        
        eval_env = gym.make(self.args.env)
        eval_env.seed(self.args.seed + 100)
        avg_reward = 0.
        
        for _ in range(eval_episodes):
                state, done = eval_env.reset(), False
                while not done:
                        action = self.actor.select_action(np.array(state))
                        state, reward, done, _ = eval_env.step(action)
                        avg_reward += reward

        avg_reward /= eval_episodes

        return avg_reward


