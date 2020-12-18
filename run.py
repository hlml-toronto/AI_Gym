from rl_class import HLML_RL
import algos.vpg.core as core
import gym
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


class myActor(core.Actor):
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


class myCritic(nn.Module):
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
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.


if __name__ == '__main__':
    input = {'training_alg': 'vpg',
             'model_list': None,
             'env': 'LunarLander-v2'}

    test_env = gym.make(input['env'])
    obs_dim = test_env.observation_space.shape[0]
    act_dim = test_env.action_space.n
    pi = myActor(obs_dim, act_dim, (64, 64), nn.Tanh)
    v = myCritic(obs_dim, (32, 32),  nn.Tanh)

    input['model_list'] = [pi, v]

    training_setup = HLML_RL(**input)
    #training_setup.train()

    training_setup.render(save=True, show=False)
