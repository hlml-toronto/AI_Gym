from presets import outdir_from_preset, PRESETS
from rl_class import HLML_RL
import pandas as pd
import matplotlib.pyplot as plt
import os


def benchmark(algorithm, env):
    """ Benchmark the specified algorithm against LunardLander-v2 with one
    thread. A fully trained model should score between 100 and 140 on a single
    run of the game.
    """
    # All algorithms are benchmarked with one thread
    user_input = {'training_alg': algorithm,
                  'env_str': env,
                  'ncpu': 1,
                  'run_name': 'benchmark',
                  'use_custom': False,
                  'actorCritic': None}

    train_input = PRESETS[algorithm]
    train_input['epochs'] = 5000

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
    algorithm = 'vpg'
    env = 'LunarLander-v2'

    # run training, evaluate trainined model
    benchmark(algorithm, env)
