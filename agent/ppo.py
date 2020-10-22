# Adjusted from https://github.com/openai/spinningup
import ray
from agent import _core
import gym,torch,random
import numpy as np
from utils.memory import PPOBuffer
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@ray.remote(num_gpus=0.5)
class PPO(object):
    def __init__(self,
        args,
        actor_critic=_core.ActorCritic_PPO,
        ac_kwargs=dict(),
        seed=0,
        steps_per_epoch=1000,
        epochs=10,
        gamma=0.99,
        pi_lr=3e-4,
        vf_lr=1e-3,
        train_pi_iters=80,
        train_v_iters=80,
        lam=0.97,
        clip_ratio=0.2,
        max_ep_len=1000,
        logger_kwargs=dict(),
        save_freq=10,
        target_kl=0.01):
        self.env = gym.make(args.env)
        self.env.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

        self.max_ep_len = max_ep_len
        self.train_v_iters = train_v_iters
        self.train_pi_iters = train_pi_iters
        self.args = args
        self.potential_max = -np.inf
        self.target_kl=target_kl
        self.clip_ratio = clip_ratio

        self.epochs = epochs
        self.local_steps_per_epoch = steps_per_epoch
        self.steps_per_epoch = steps_per_epoch

        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]

        ac_kwargs['action_space'] = self.env.action_space
        self.actor_critic = actor_critic(in_features=obs_dim,**ac_kwargs)
        self.buf = PPOBuffer(obs_dim, act_dim, self.local_steps_per_epoch, gamma=gamma, lam=lam)

        # Count variables
        var_counts = tuple(_core.count_vars(module) for module in
        [self.actor_critic.policy, self.actor_critic.vf_mlp])
        print('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

        self.train_pi = torch.optim.Adam(self.actor_critic.policy.parameters(), lr=pi_lr)
        self.train_v = torch.optim.Adam(self.actor_critic.vf_mlp.parameters(), lr=vf_lr)
    
    def get_actor_params(self):
        self.actor_critic.policy.cpu()
        return self.actor_critic.policy.net.state_dict(),self.actor_critic.policy.mu.state_dict()


    def transfer(self,sac_params_net,sac_params_mu,sac_value_params):
        self.actor_critic.policy.net.load_state_dict(sac_params_net)
        self.actor_critic.policy.mu.load_state_dict(sac_params_mu)
        self.actor_critic.vf_mlp.load_state_dict(sac_value_params)


    def update(self):
        self.actor_critic.train()
        self.actor_critic.cuda()

        obs, act, adv, ret, logp_act, act_old = [torch.Tensor(x) for x in self.buf.get()]

        obs = torch.FloatTensor(obs).to(device)
        act = torch.FloatTensor(act).to(device)
        act_old = torch.FloatTensor(act_old).to(device)
        adv = torch.FloatTensor(adv).to(device)
        ret = torch.FloatTensor(ret).to(device)
        logp_a_old = torch.FloatTensor(logp_act).to(device)

        _, _, _, logp_a,_ = self.actor_critic.policy(obs, act_old)

        ratio = (logp_a - logp_a_old).exp()
        min_adv = torch.where(adv > 0, (1 + self.clip_ratio) * adv,
                              (1 - self.clip_ratio) * adv)
        pi_l_old = -(torch.min(ratio * adv, min_adv)).mean()
        ent = (-logp_a).mean()

        for i in range(self.train_pi_iters):

            _, _, _, logp_a,_ = self.actor_critic.policy(obs, act_old)

            ratio = (logp_a - logp_a_old).exp()
            min_adv = torch.where(adv > 0, (1 + self.clip_ratio) * adv,
                                  (1 - self.clip_ratio) * adv)
            pi_loss = -(torch.min(ratio * adv, min_adv)).mean()

            # Policy gradient step
            self.train_pi.zero_grad()
            pi_loss.backward()
            self.train_pi.step()

            _, _, _, logp_a, _ = self.actor_critic.policy(obs, act_old)
            kl = (logp_a_old - logp_a).mean()
            if kl > 1.5 * self.target_kl:
                break

        v = self.actor_critic.vf_mlp(obs)
        v_l_old = F.mse_loss(v, ret)
        for _ in range(self.train_v_iters):
            v = self.actor_critic.vf_mlp(obs)
            v_loss = F.mse_loss(v, ret)

            self.train_v.zero_grad()
            v_loss.backward()
            self.train_v.step()

    def train(self,iteraion,return_gap,sac_ppo):
        ppo_timesteps = 0
        global_mem_temp = []
        local_mem_temp = []
        one_epsoide = []

        o, r, d, ep_ret, ep_len = self.env.reset(), 0, False, 0, 0

        while ppo_timesteps < iteraion:
            self.actor_critic.eval()
            self.actor_critic.cpu()
            for t in range(self.local_steps_per_epoch):
                a_scaled,logp_pi, a, _, v_t = self.actor_critic(torch.Tensor(o.reshape(1,-1))) # pi, logp_pi, v, logp

                self.buf.store(o, a_scaled.detach().numpy(), r, v_t.item(), logp_pi.detach().numpy(),a.detach().numpy())

                old_o = o
                o, r, d, _ = self.env.step(a_scaled.detach().numpy()[0])
                ep_ret += r
                ep_len += 1
                ppo_timesteps += 1

                one_epsoide.append((old_o, a_scaled.detach().numpy()[0], r, o, d))

                terminal = d or (ep_len == self.max_ep_len)

                if terminal or (t == (self.local_steps_per_epoch-1)):
                    # if trajectory didn't reach terminal state, bootstrap value target
                    last_val = r if d else self.actor_critic.vf_mlp(torch.Tensor(o.reshape(1,-1))).item()
                    self.buf.finish_path(last_val)
                    global_mem_temp.append(one_epsoide)

                    if sac_ppo:
                        if ep_ret > return_gap:
                            local_mem_temp.append(one_epsoide)
                    o, r, d, ep_ret, ep_len = self.env.reset(), 0, False, 0, 0
                    one_epsoide = []

            self.update()

        return ppo_timesteps,global_mem_temp,local_mem_temp

    def test_policy(self, eval_episodes=5):
        self.actor_critic.policy.cpu()
        
        eval_env = gym.make(self.args.env)
        eval_env.seed(self.args.seed + 100)
        avg_reward = 0.
        
        for _ in range(eval_episodes):
                state, done = eval_env.reset(), False
                while not done:
                        action = self.actor_critic.policy.select_action(np.array(state))
                        state, reward, done, _ = eval_env.step(action)
                        avg_reward += reward

        avg_reward /= eval_episodes

        return avg_reward


