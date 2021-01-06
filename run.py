import os

from benchmark import plot_performance_timeseries
from customActorCritic import customActorCritic, TRAINING_ALG, CUSTOM_AC_DEFAULT_KWARGS
from presets import outdir_from_preset
from rl_class import HLML_RL


def main(user_input, train_input):
    # specify output directory (if not restarting a run)
    if user_input['restart_tuple'] is None:
        default_out = outdir_from_preset(user_input['training_alg'], user_input['use_custom'], user_input['env_str'])
        if user_input['run_name'] is not None:
            default_out = default_out + '_' + user_input['run_name']
        user_input['exp_name'] = default_out
    else:
        # assumes user_input['restart_tuple'] is two-tuple of the form (exp_name, exp_name_detailed)
        assert len(user_input['restart_tuple']) == 2
        exp_name_original = user_input['restart_tuple'][0]
        exp_name_original_seed = user_input['restart_tuple'][1]  # should be of the form '%s_s%d % (exp_name, seed)'
        exp_name_restart = exp_name_original + '_restart'
        user_input['exp_name'] = exp_name_restart

    # build custom actor critic nn modules
    if user_input['use_custom']:
        user_input['actorCritic'] = customActorCritic
    else:
        user_input['actorCritic'] = None

    # enforce user input seed for training kwargs (defaults to 0)
    train_input['seed'] = user_input.get('seed', 0)

    # train the model
    training_setup = HLML_RL(**user_input)
    if user_input['use_custom']:
        print("The user's custom ActorCritic class should adhere to the following documentation\n\n\n")
        training_setup.ac_help()  # print the documentation for a custom ActorCritic

    if user_input['restart_tuple'] is None:
        training_setup.train(**train_input)
        out_dir = os.path.join('experiments', default_out, default_out + '_s%d' % training_setup.seed)
        progress = 'progress.txt'
        timeseries = 'training_performance.png'
    else:
        training_setup.restart_train(**train_input)
        out_dir = os.path.join('experiments', exp_name_original, exp_name_original_seed)
        progress = 'progress_restart.txt'
        timeseries = 'training_performance_restart.png'

    # plot model performance timeseries
    plot_performance_timeseries(out_dir, user_input['training_alg'], user_input['env_str'],
                                progress_file=progress, out=timeseries)

    # render the trained model
    training_setup.render(save=True, show=False, seed=training_setup.seed,
                          pytsave_path=out_dir + os.sep + 'pyt_save')


if __name__ == '__main__':
    # Environments to choose from:  https://github.com/openai/gym/wiki/Table-of-environments (box = continuous)
    #  - 'LunarLander-v2'            (discrete action space)
    #  - 'MountainCarContinuous-v0'  (continuous action space)
    #  - 'BipedalWalker-v3'          (continuous action space)       - Hard

    # general user choices
    #  - training_alg:            (str) specified in custom_ActorCritic.py (e.g. 'vpg' or 'ddpg')
    #  - use_custom:              (Bool) choose algorithm variant: custom or not
    #  - env_str:                 (str) e.g. 'LunarLander-v2' or 'MountainCarContinuous-v0'
    #  - training_alg:            (str) choose a training algoirthm (e.g. 'vpg')
    #  - ncpu:                    (int) MPI multithreading (some algos will not support)  # TODO gpu flag arg as well?
    #  - seed:                    (int) run seed (also affects output directory)
    #  - restart_tuple:           (tuple) if not None, then restart training using last saved epoch state
    #                               assumes a two-tuple of the form (exp_name, exp_name_s%d)
    #  - run_name:                (str) if not None, append custom name for your training run (ignored if restarting)
    user_input = {'training_alg': TRAINING_ALG,
                  'use_custom': True,
                  'env_str': 'LunarLander-v2',
                  'ncpu': 1,
                  'seed': 8,
                  'run_name': None,
                  'restart_tuple': ('vpg_HLML_LunarLander-v2', 'vpg_HLML_LunarLander-v2_s8')}

    # detailed user choices: specify overrides for algorithm defaults found in master dict PRESETS in presets.py
    train_input = {}
    if user_input['use_custom']:
        train_input = {'ac_kwargs': CUSTOM_AC_DEFAULT_KWARGS}

    # change any training hyperparameters at runtime, e.g. for fine tuning your hyperparameters from defaults
    #  - save_freq:             (int) save model to file every N epochs (defaults to None if not set)
    train_input.update({'epochs': 3,
                        'save_freq': 1,
                        'render_saves': True})

    # run training, visualize trained model
    main(user_input, train_input)
