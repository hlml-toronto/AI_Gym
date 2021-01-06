import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

import algos.vpg.core as core
from rl_class import get_IO_dim

# ------------------------------------------------------------------------------
# This file is where a custom ActorCritic can be defined by the user to be used
# in RL training in run.py
# ------------------------------------------------------------------------------

# specify the name of the training algorithm for your custom ActorCritic
TRAINING_ALG = 'vpg'

# write default hyperparameters for your custom ActorCritic
CUSTOM_AC_DEFAULT_KWARGS = {'ac_kwargs' : {'hidden_sizes' : (6,6,6) } } }


class customActor(core.Actor):
    """
    Custom user policy model
    """
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        output_activation = nn.Identity
        layers = []
        for j in range(len(sizes)-1):
            act = activation if j < len(sizes)-2 else output_activation
            layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
        self.logits_net = nn.Sequential(*layers)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


# TODO determine if these are VPG specific; move if so
class customCritic(nn.Module):
    """
    Custom user on-policy value function model
    """
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        sizes = [obs_dim] + list(hidden_sizes) + [1]
        output_activation = nn.Identity
        layers = []
        for j in range(len(sizes)-1):
            act = activation if j < len(sizes)-2 else output_activation
            layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
        self.v_net = nn.Sequential(*layers)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)  # Critical to ensure v has right shape.


class customActorCritic(nn.Module):
    """Sample code for a custom ActorCritic class to be used with the VPG
    training algorithm.
    """
    def __init__(self, obs_env, act_env, hidden_sizes=(64, 64),
                 activation=nn.Tanh):
        super().__init__()
        obs_dim_tuple, act_dim_tuple = get_IO_dim((obs_env, act_env))
        obs_dim = obs_dim_tuple[0]; act_dim = act_dim_tuple[0]
        # policy builder depends on action space
        self.pi = customActor(obs_dim, act_dim, hidden_sizes, activation)

        # build value function
        self.v = customCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]
