# Some code are adjusted from https://github.com/openai/spinningup and https://github.com/apourchot/CEM-RL/blob/master/models.py
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from gym.spaces import Box, Discrete
from copy import deepcopy
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence
from torch.optim import Adam, SGD
from torch.autograd import Variable
from torch.distributions import Uniform


LOG_STD_MAX = 2
LOG_STD_MIN = -20
EPS = 1e-6


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = torch.cuda.is_available()

def to_numpy(var):
    return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()


def count_vars(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

class RLNN(nn.Module):

    def __init__(self):
        super(RLNN, self).__init__()

    def set_params(self, params):
        """
        Set the params of the network to the given parameters
        """
        cpt = 0
        for param in self.parameters():
            tmp = np.product(param.size())

            if torch.cuda.is_available():
                param.data.copy_(torch.from_numpy(
                    params[cpt:cpt + tmp]).view(param.size()).cuda())
            else:
                param.data.copy_(torch.from_numpy(
                    params[cpt:cpt + tmp]).view(param.size()))
            cpt += tmp

    def get_params(self):
        """
        Returns parameters of the actor
        """
        return deepcopy(np.hstack([to_numpy(v).flatten() for v in
                                   self.parameters()]))

    def get_grads(self):
        """
        Returns the current gradient
        """
        return deepcopy(np.hstack([to_numpy(v.grad).flatten() for v in self.parameters()]))

    def get_size(self):
        """
        Returns the number of parameters of the network
        """
        return self.get_params().shape[0]

    def load_model(self, filename, net_name):
        """
        Loads the model
        """
        if filename is None:
            return

        self.load_state_dict(
            torch.load('{}/{}.pkl'.format(filename, net_name),
                       map_location=lambda storage, loc: storage)
        )

    def save_model(self, output, net_name):
        """
        Saves the model
        """
        torch.save(
            self.state_dict(),
            '{}/{}.pkl'.format(output, net_name)
        )


class MLP(nn.Module):
    def __init__(self,
                layers,
                activation=torch.tanh,
                output_activation=None,
                output_scale=1,
                output_squeeze=False):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.output_activation = output_activation
        self.output_scale = output_scale
        self.output_squeeze = output_squeeze

        for i, layer in enumerate(layers[1:]):
            self.layers.append(nn.Linear(layers[i], layer))
            nn.init.zeros_(self.layers[i].bias)

    def forward(self, inputs):
        x = inputs
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        if self.output_activation is None:
            x = self.layers[-1](x) * self.output_scale
        else:
            x = self.output_activation(self.layers[-1](x)) * self.output_scale
        return x.squeeze() if self.output_squeeze else x


# Based on policy gradient, policy sample
class BasicGaussianPolicy(RLNN):
    """docstring for BasicGaussianPolicy"""
    def __init__(self, in_features, hidden_sizes, activation,output_activation, action_space):
        super(BasicGaussianPolicy, self).__init__()
        self.action_dim = action_space.shape[0]
        self.action_scale = action_space.high[0]
        self.output_activation = output_activation

        self.net = MLP(
            layers=[in_features] + list(hidden_sizes),
            activation=activation,
            output_activation=activation)

        self.mu = nn.Linear(
            in_features=list(hidden_sizes)[-1], out_features=self.action_dim)


class GaussianPolicy_SAC(BasicGaussianPolicy):
    def __init__(self, in_features, hidden_sizes, activation,
                 output_activation, action_space):
        super(GaussianPolicy_SAC, self).__init__(in_features, hidden_sizes, activation,
                 output_activation, action_space)

        self.log_std = nn.Sequential(
            nn.Linear(
                in_features=list(hidden_sizes)[-1], out_features=self.action_dim),
            nn.Tanh())

    def forward(self, x):
        output = self.net(x)
        mu = self.mu(output)
        
        if self.output_activation:
            mu = self.output_activation(mu)

        log_std = self.log_std(output)

        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            log_std + 1)

        policy = Normal(mu, torch.exp(log_std))

        pi = policy.rsample()
        logp_pi = torch.sum(policy.log_prob(pi), dim=1)

        mu, pi, logp_pi = self._apply_squashing_func(mu, pi, logp_pi)        # make sure actions are in correct range
        
        mu_scaled = mu * self.action_scale
        pi_scaled = pi * self.action_scale


        return pi_scaled,mu_scaled,logp_pi   # pi_scaled, mu_scaled, logp_pi -- sac

    def _clip_but_pass_gradient(self, x, l=-1., u=1.):
        clip_up = (x > u).float()
        clip_low = (x < l).float()
        return x + ((u - x) * clip_up + (l - x) * clip_low).detach()

    def _apply_squashing_func(self, mu, pi, logp_pi):
        mu = torch.tanh(mu)
        pi = torch.tanh(pi)

        # To avoid evil machine precision error, strictly clip 1-pi**2 to [0,1] range.
        logp_pi -= torch.sum(
            torch.log(self._clip_but_pass_gradient(1 - pi**2, l=0, u=1) + EPS),
            dim=1)

        return mu, pi, logp_pi

    def select_action(self,o,_eval=False):
        pi, mu, _ = self.forward(torch.Tensor(o.reshape(1, -1)))
        return mu.cpu().detach().numpy()[0] if _eval else pi.cpu().detach().numpy()[0]


class ActorCritic_SAC(nn.Module):
    def __init__(self,
                in_features,
                action_space,
                hidden_sizes=(400, 300),
                activation=torch.relu,
                output_activation=None,
                policy=GaussianPolicy_SAC):
        super(ActorCritic_SAC, self).__init__()

        action_dim = action_space.shape[0]

        self.policy = policy(in_features, hidden_sizes, activation,
                            output_activation, action_space)

        self.vf_mlp = MLP(
            [in_features] + list(hidden_sizes) + [1],
            activation,
            output_squeeze=True)

        self.q1 = MLP(
            [in_features + action_dim] + list(hidden_sizes) + [1],
            activation,
            output_squeeze=True)

        self.q2 = MLP(
            [in_features + action_dim] + list(hidden_sizes) + [1],
            activation,
            output_squeeze=True)

    def forward(self, x, a):
        pi, mu, logp_pi= self.policy(x)

        q1 = self.q1(torch.cat((x, a), dim=1))
        q1_pi = self.q1(torch.cat((x, pi), dim=1))

        q2 = self.q2(torch.cat((x, a), dim=1))
        q2_pi = self.q2(torch.cat((x, pi), dim=1))

        v = self.vf_mlp(x)

        return pi, mu, logp_pi, q1, q2, q1_pi, q2_pi, v

 
class GaussianPolicy_PPO(BasicGaussianPolicy):
    def __init__(self, in_features, hidden_sizes, activation,
                 output_activation, action_space):
        super(GaussianPolicy_PPO, self).__init__(in_features, hidden_sizes, activation,
                 output_activation, action_space)

        self.log_std = nn.Parameter(-0.5 * torch.ones(self.action_dim))


    def forward(self, x, a=None):
        output = self.net(x)
        mu = self.mu(output)
        if self.output_activation:
            mu = self.output_activation(mu)

        policy = Normal(mu, self.log_std.exp())

        pi = policy.sample() ## vpg
        pi_old = pi

        logp_pi = policy.log_prob(pi).sum(dim=1)
        
        if a is not None:
            logp_a = policy.log_prob(a).sum(dim=1)
        else:
            logp_a = None

        mu, pi, logp_pi,logp_a = self._apply_squashing_func(mu, pi, logp_pi,a,logp_a)
        # # make sure actions are in correct range
        mu_scaled = mu * self.action_scale
        pi_scaled = pi * self.action_scale

        return pi_scaled,mu_scaled,logp_pi,logp_a,pi_old 

    def _clip_but_pass_gradient(self, x, l=-1., u=1.):
        clip_up = (x > u).float()
        clip_low = (x < l).float()
        return x + ((u - x) * clip_up + (l - x) * clip_low).detach()

    def _apply_squashing_func(self, mu, pi, logp_pi, a, logp_a):
        mu = torch.tanh(mu)
        pi = torch.tanh(pi)
        
        if a is not None:
            a = torch.tanh(a)

        # To avoid evil machine precision error, strictly clip 1-pi**2 to [0,1] range.
        logp_pi -= torch.sum(
            torch.log(self._clip_but_pass_gradient(1 - pi**2, l=0, u=1) + EPS),
            dim=1)

        if a is not None:
            logp_a -= torch.sum(
                torch.log(self._clip_but_pass_gradient(1 - a**2, l=0, u=1) + EPS),
                dim=1)

        return mu, pi, logp_pi, logp_a

    def select_action(self,o,_eval=False):
        pi_scaled,mu_scaled, _, _, _ = self.forward(torch.Tensor(o.reshape(1, -1)))
        return mu_scaled.cpu().detach().numpy()[0] if _eval else pi_scaled.cpu().detach().numpy()[0]


# include policy and value function
class ActorCritic_PPO(nn.Module):
    def __init__(self, in_features, action_space,
                 hidden_sizes=(400, 300), activation=torch.relu,
                 output_activation=None, policy=None):
        super(ActorCritic_PPO, self).__init__()

        self.policy = GaussianPolicy_PPO(in_features, hidden_sizes,
                                     activation, output_activation,
                                     action_space) 
    
        self.vf_mlp = MLP(layers=[in_features]+list(hidden_sizes)+[1],
                                  activation=activation, output_squeeze=True)

    def forward(self, x, a=None):
        pi_scaled,mu_scaled,logp_pi,logp_a, a = self.policy(x, a)
        v = self.vf_mlp(x)

        return pi_scaled, logp_pi,a,logp_a,v


class Deterministic_CEM(BasicGaussianPolicy):
    def __init__(self, in_features, hidden_sizes, activation,
                 output_activation, action_space):
        super(Deterministic_CEM, self).__init__(in_features, hidden_sizes, activation,
                 output_activation, action_space)

    def forward(self, x):
        output = self.net(x)
        mu = self.mu(output)
        mu = torch.tanh(mu)

        return mu

    def select_action(self,o,_eval=True):
        mu = self.forward(torch.Tensor(o.reshape(1, -1)))
        return np.clip(mu.cpu().detach().numpy()[0], -self.action_scale, self.action_scale)
