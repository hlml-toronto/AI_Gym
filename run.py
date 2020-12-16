from rl_class import HLML_RL
from algos.vpg.core import MLPCategoricalActor, MLPCritic
import gym
import torch.nn as nn


if __name__ == '__main__':
    input = {'training_alg': 'vpg',
             'model_list': None,
             'env': 'LunarLander-v2'}

    test_env = gym.make(input['env'])
    obs_dim = test_env.observation_space.shape[0]
    act_dim = test_env.action_space.n
    pi = MLPCategoricalActor(obs_dim, act_dim, (64, 64), nn.Tanh)
    v = MLPCritic(obs_dim, (32, 32),  nn.Tanh)

    input['model_list'] = [pi, v]

    training_setup = HLML_RL(**input)
    training_setup.train()
