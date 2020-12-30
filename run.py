import gym
import torch
import torch.nn as nn
from gym.spaces import Box, Discrete
from torch.distributions.categorical import Categorical

import algos.vpg.core as core
from presets import outdir_from_preset
from rl_class import HLML_RL, get_IO_dim


# Environments to choose from:  https://github.com/openai/gym/wiki/Table-of-environments (box = continuous)
#  - 'LunarLander-v2'            (discrete action space)
#  - 'MountainCarContinuous-v0'  (continuous action space)
#  - 'BipedalWalker-v3'          (continuous action space)       - Hard


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


class myActorCritic(nn.Module):
    """Sample code for a custom ActorCritic class to be used with the VPG
    training algorithm.
    """
    def __init__(self, obs_env, act_env, hidden_sizes=(64, 64),
                 activation=nn.Tanh):
        super().__init__()
        obs_dim, act_dim = get_IO_dim((obs_env, act_env))
        # policy builder depends on action space
        self.pi = myActor(obs_dim, act_dim, hidden_sizes, activation)

        # build value function
        self.v = myCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]


if __name__ == '__main__':
    # general user choices
    #  - training_alg:            (str) choose a training algorithm (e.g. 'vpg' or 'ddpg')
    #  - training_alg_variant:    (str) choose algorithm variant: 'HLML' or 'spinningup'
    #  - env_str:                 (str) e.g. 'LunarLander-v2' or 'MountainCarContinuous-v0'
    #  - training_alg:            (str) choose a training algoirthm (e.g. 'vpg')
    #  - ncpu:                    (int) MPI multithreading (some algos will not support)  # TODO gpu flag arg as well?
    #  - run_name:                (str) custom name for your training run
    user_input = {'training_alg': 'vpg',
                  'training_alg_variant': 'HLML',
                  'env_str': 'LunarLander-v2',
                  'ncpu': 1,
                  'run_name': None}

    # detailed user choices: specify overrides for algorithm defaults found in master dict PRESETS in presets.py
    train_input = {'epochs': 1,
                   'steps_per_epoch': 1000}

    # specify output directory
    default_out = outdir_from_preset(user_input['training_alg'], user_input['training_alg_variant'])
    default_out += '_' + user_input['env_str']  # include env in out dir
    if user_input['run_name'] is not None:
        user_out = default_out + '_' + user_input['run_name']
    user_input['exp_name'] = default_out

    # build custom actor critic nn modules
    if user_input['training_alg_variant'] == 'HLML':
        # TODO maybe move this out, see comments in presets.py
        user_input['actorCritic'] = myActorCritic
    else:
        user_input['actorCritic'] = None

    # train the model
    training_setup = HLML_RL(**user_input)
    if user_input['training_alg_variant'] == 'HLML':
        print("The user's custom ActorCritic class should adhere to the following documentation\n\n\n")
        training_setup.ac_help()  # print the documentation for a custom ActorCritic
    training_setup.train(**train_input)

    # render the trained model
    training_setup.render(save=True, show=False)
