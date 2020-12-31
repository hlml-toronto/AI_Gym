import gym
import torch
import torch.nn as nn
from gym.spaces import Box, Discrete
from torch.distributions.categorical import Categorical

import algos.vpg.core as core
from presets import outdir_from_preset
from rl_class import HLML_RL, get_IO_dim
from custom_ActorCritic import customActorCritic


def main(user_input, train_input):
    # specify output directory
    default_out = outdir_from_preset(user_input['training_alg'], user_input['training_alg_variant'])
    default_out += '_' + user_input['env_str']  # include env in out dir
    if user_input['run_name'] is not None:
        default_out = default_out + '_' + user_input['run_name']
    user_input['exp_name'] = default_out

    # build custom actor critic nn modules
    if user_input['training_alg_variant'] == 'HLML':
        user_input['actorCritic'] = customActorCritic
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


if __name__ == '__main__':
    # Environments to choose from:  https://github.com/openai/gym/wiki/Table-of-environments (box = continuous)
    #  - 'LunarLander-v2'            (discrete action space)
    #  - 'MountainCarContinuous-v0'  (continuous action space)
    #  - 'BipedalWalker-v3'          (continuous action space)       - Hard


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

    # run training, visualize trainined model
    main(user_input, train_input)
