from presets import outdir_from_preset, PRESETS
from rl_class import HLML_RL
import pandas as pd
import matplotlib.pyplot as plt
import os

"""
Notes:
- ncpu with ddpg: appears detrimental
- ncpu with vpg: setting ncpu 5 or 6 gave ~2.5x speedup on LunarLander-v2
- ddpg epochs are much slower than vpg (for 'BipedalWalker-v3'): 60s vs 4.2s per epoch

Issues:
- 'BipedalWalker-v3' + vpg appears to hang with ncpu > 1
- 'CarRacing-v0' + vpg fails quickly with torch error
- 'CarRacing-v0' + ddpg fails quickly with memory error
 
Tested:
- vpg + ncpu>=1 : 'LunarLander-v2', 'CartPole-v1', 'Pendulum-v0', 'Acrobot-v1'
- vpg + npu=1 : 'BipedalWalker-v3'
- ddpg: 'MountainCarContinuous-v0', 'BipedalWalker-v3'
"""


def benchmark(algorithm, env):
    """ Benchmark the specified algorithm against env with one thread
    """
    # All algorithms are benchmarked with one thread
    user_input = {'training_alg': algorithm,
                  'env_str': env,
                  'ncpu': 1,
                  'run_name': 'benchmark',
                  'use_custom': False,
                  'actorCritic': None}

    train_input = PRESETS[algorithm]
    # The choice of epochs was found empirically
    if env == 'MountainCarContinuous-v0':
        train_input['epochs'] = 100
    elif env == 'LunarLander-v2':
        train_input['epochs'] = 5000
    else:
        print('Warning: untested environment specified')
        train_input['epochs'] = 20

    # specify output directory
    default_out = outdir_from_preset(algorithm, False, env) + '_' + 'benchmark'
    user_input['exp_name'] = default_out

    # train the model
    bench = HLML_RL(**user_input)
    bench.train(**train_input)

    # benchmark the trained model
    log_out = os.path.join('experiments', default_out, default_out + '_s' + str(bench.seed))
    fname = log_out + os.sep + 'progress.txt'
    df = pd.read_csv(fname, delim_whitespace=True)

    fig, axes = plt.subplots(nrows=1, ncols=1)
    x = df['Time']
    y = df['AverageEpRet']
    yup = df['AverageEpRet'] + df['StdEpRet']
    ydown = df['AverageEpRet'] - df['StdEpRet']
    plt.plot(x, y, 'k--')
    plt.fill_between(x, ydown, yup, alpha=.3)
    plt.title(user_input['training_alg'] + ':  ' + env)
    plt.xlabel('Wall time')
    plt.ylabel('Average Epoch Reward')
    plt.savefig(log_out + os.sep + 'benchmark.png')


if __name__ == '__main__':
    # These are the only lines that the user should change:
    algorithm = 'sac'
    env = 'MountainCarContinuous-v0'

    # run training, evaluate trained model
    benchmark(algorithm, env)
