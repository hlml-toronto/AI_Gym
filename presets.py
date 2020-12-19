import gym

DEFAULT_ENV = 'LunarLander-v2'

# Define default parameters for each implemented training algorithm
DEFAULT_KWARGS = {'vpg':
                      {'env': lambda: gym.make(DEFAULT_ENV),
                       'ac_kwargs': dict(hidden_sizes=[64] * 2),
                       'gamma': 0.99,
                       'seed': 0,
                       'cpu': 1,
                       'steps': 4000,
                       'epochs': 50,
                       'exp_name': 'vpg'},
                  'ddpg':
                      {'env': lambda: gym.make(DEFAULT_ENV),
                       'ac_kwargs': dict(hidden_sizes=[64] * 2),
                       'gamma': 0.99,
                       'seed': 0,
                       'cpu': 1,
                       'steps': 4000,
                       'epochs': 50,
                       'exp_name': 'ddpg'}
                  }

IMPLEMENTED_ALGOS = DEFAULT_KWARGS.keys()

# TODO
"""  
if vpg do
    import vpgstuff
    ...
elif
    ...
else
    raise not implemented"""
