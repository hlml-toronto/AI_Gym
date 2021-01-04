from presets import outdir_from_preset, PRESETS
from rl_class import HLML_RL
import pandas as pd
import matplotlib.pyplot as plt
import os

"""
Notes:
- ncpu > 1 with vpg:  setting ncpu 5 or 6 gave ~2.5x speedup on LunarLander-v2
- ncpu > 1 with ddpg: appears detrimental
- ncpu > 1 with sac:  appears detrimental
- ncpu > 1 with ppo:  fails with ncpu > 2 for 'MountainCarContinuous-v0', but works for 'Pendulum-v0'
- ncpu > 1 with td3:  appears detrimental
- ddpg epochs are much slower than vpg (for 'BipedalWalker-v3'): 60s vs 4.2s per epoch

Issues:
- 'BipedalWalker-v3' + vpg           appears to hang with ncpu > 1
- 'MountainCarContinuous-v0' + ppo   crash when ncpu > 2
- 'CarRacing-v0' + vpg               fails quickly with torch error
- 'CarRacing-v0' + ddpg              fails quickly with memory error
- 'CarRacing-v0' + sac               fails quickly with memory error
- 'CarRacing-v0' + ppo               fails quickly with torch error
- 'CarRacing-v0' + td3               fails quickly with memory error

Tested:
- vpg  + ncpu>=1 : 'LunarLander-v2', 'CartPole-v1', 'Pendulum-v0', 'Acrobot-v1'
- vpg  + npu=1 :   'BipedalWalker-v3'
- ddpg + npu=1 :   'MountainCarContinuous-v0'
- sac  + npu=1 :   'MountainCarContinuous-v0'
- ppo  + npu=1 :   'MountainCarContinuous-v0'
- td3  + npu=1 :   'MountainCarContinuous-v0'
"""


def plot_performance_timeseries(log_out, training_alg, env):
    # read progress.txt
    fname = log_out + os.sep + 'progress.txt'
    progress_df = pd.read_csv(fname, delim_whitespace=True)
    # plot
    fig, axes = plt.subplots(nrows=1, ncols=1)
    x = progress_df['Time']
    y = progress_df['AverageEpRet']
    yup = progress_df['AverageEpRet'] + progress_df['StdEpRet']
    ydown = progress_df['AverageEpRet'] - progress_df['StdEpRet']
    plt.plot(x, y, 'k--')
    plt.fill_between(x, ydown, yup, alpha=.3)
    plt.title(training_alg + ':  ' + env)
    plt.xlabel('Wall time')
    plt.ylabel('Average Epoch Reward')
    plt.savefig(log_out + os.sep + 'training_performance.png')
    return


def benchmark(algorithm, env):
    """ Benchmark the specified algorithm against env with one thread
    """
    # All algorithms are benchmarked with one thread
    user_input = {'training_alg': algorithm,
                  'env_str': env,
                  'ncpu': 2,
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

    # plot model performance timeseries
    log_out = os.path.join('experiments', default_out, default_out + '_s' + str(bench.seed))
    plot_performance_timeseries(log_out, algorithm, env)


if __name__ == '__main__':
    # These are the only lines that the user should change:
    algorithm = 'ppo'
    env = 'MountainCarContinuous-v0'

    # run training, evaluate trained model
    benchmark(algorithm, env)
