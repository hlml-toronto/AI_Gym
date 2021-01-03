from presets import outdir_from_preset, DEFAULT_ENV_STR
from rl_class import HLML_RL
from customActorCritic import customActorCritic, TRAINING_ALG, CUSTOM_AC_DEFAULT_KWARGS


def main(user_input, train_input):
    # specify output directory
    default_out = outdir_from_preset(user_input['training_alg'], user_input['use_custom'], user_input['env_str'])
    if user_input['run_name'] is not None:
        default_out = default_out + '_' + user_input['run_name']
    user_input['exp_name'] = default_out

    # build custom actor critic nn modules
    if user_input['use_custom']:
        user_input['actorCritic'] = customActorCritic
    else:
        user_input['actorCritic'] = None

    # train the model
    training_setup = HLML_RL(**user_input)
    if user_input['use_custom']:
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
    #  - training_alg:            (str) specified in custom_ActorCritic.py (e.g. 'vpg' or 'ddpg')
    #  - use_custom:              (Bool) choose algorithm variant: custom or not
    #  - env_str:                 (str) e.g. 'LunarLander-v2' or 'MountainCarContinuous-v0'
    #  - training_alg:            (str) choose a training algoirthm (e.g. 'vpg')
    #  - ncpu:                    (int) MPI multithreading (some algos will not support)  # TODO gpu flag arg as well?
    #  - run_name:                (str) custom name for your training run
    user_input = {'training_alg': TRAINING_ALG,
                  'use_custom': False,
                  'env_str': 'LunarLander-v2',
                  'ncpu': 1,
                  'run_name': None}

    # detailed user choices: specify overrides for algorithm defaults found in master dict PRESETS in presets.py
    train_input = CUSTOM_AC_DEFAULT_KWARGS

    # change any training hyperparameters at runtime, e.g. for fine tuning your hyperparameters from defaults
    train_input.update(
                       {'epochs': 1,
                        'steps_per_epoch': 1000})

    # run training, visualize trainined model
    main(user_input, train_input)
