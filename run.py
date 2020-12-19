import gym
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

import algos.vpg.core as core
from presets import outdir_from_preset
from rl_class import HLML_RL


# TODO determine if these are VPG specific; move if so
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


# TODO determine if these are VPG specific; move if so
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
        return torch.squeeze(self.v_net(obs), -1)  # Critical to ensure v has right shape.


if __name__ == '__main__':
    # general user choices
    user_input = {'training_alg': 'vpg',             # choose a training algoirthm (e.g. 'vpg')
                  'training_alg_variant': 'HLML',    # choose 'HLML' or 'spinningup' algorithm variant
                  'env_str': 'LunarLander-v2',       # choose an experiment
                  'ncpu': 1,                         # MPI multithreading (some algos will not support)  # TODO gpu argument ?
                  'run_name': None}                  # str: custom name for your training run

    # detailed user choices: specify overrides for the defaults found in master dict PRESETS in presets.py
    train_input = {'epochs': 1}

    # specify output directory
    default_out = outdir_from_preset(user_input['training_alg'], user_input['training_alg_variant'])
    if user_input['run_name'] is not None:
        user_out = default_out + '_' + user_input['run_name']
    user_input['exp_name'] = default_out

    # build custom actor critic nn modules
    # TODO maybe move this out, see comments in presets.py
    test_env = gym.make(user_input['env_str'])
    obs_dim = test_env.observation_space.shape[0]
    act_dim = test_env.action_space.n
    pi = myActor(obs_dim, act_dim, (64, 64), nn.Tanh)
    v = myCritic(obs_dim, (32, 32),  nn.Tanh)
    user_input['model_list'] = [pi, v]            # specify nn modules here

    # train the model
    training_setup = HLML_RL(**user_input)
    training_setup.train(**train_input)

    # render the trained model
    training_setup.render(save=True, show=False)
