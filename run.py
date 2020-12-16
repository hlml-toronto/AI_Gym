from rl_class import HLML_RL
import gym

if __name__ == '__main__':
    input = {'training_alg': 'vpg',
             'model_list': None,
             'env': 'LunarLander-v2'}
    training_setup = HLML_RL(**input)
    training_setup.train()
