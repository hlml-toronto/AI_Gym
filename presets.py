# choose if MPI multi-threading (n > 1) or not (n = 1)
DEFAULT_NCPU = 1

# prepare the default environment
DEFAULT_ENV_STR = 'LunarLander-v2'
TESTED_ENVS = ['LunarLander-v2', 'MountainCarContinuous-v0', 'BipedalWalker-v3']

# define default parameters for each implemented training algorithm
PRESETS = {'vpg':
                {'ac_kwargs': dict(hidden_sizes=[64] * 2),
                 'gamma': 0.99,
                 'seed': 0,
                 'steps_per_epoch': 4000,
                 'epochs': 50,
                 'pi_lr': 3e-4,
                 'vf_lr': 1e-3,            # unique to vpg (not in ddpg)
                 'max_ep_len': 1000,
                 'save_freq': 10,
                 'logger_kwargs': dict(),
                 'train_v_iters': 80,      # unique to vpg (not in ddpg)
                 'lam': 0.97},             # unique to vpg (not in ddpg)

           'ddpg':
                {'ac_kwargs': dict(hidden_sizes=[256] * 2),
                 'gamma': 0.99,
                 'seed': 0,
                 'steps_per_epoch': 4000,
                 'epochs': 100,
                 'pi_lr': 1e-3,
                 'q_lr': 1e-3,              # unique to ddpg (not in vpg)
                 'max_ep_len': 1000,
                 'save_freq': 1,
                 'logger_kwargs': dict(),
                 'replay_size': int(1e6),   # unique to ddpg (not in vpg)
                 'polyak': 0.995,           # unique to ddpg (not in vpg)
                 'batch_size': 100,         # unique to ddpg (not in vpg)
                 'start_steps': 10000,      # unique to ddpg (not in vpg)
                 'update_after': 1000,      # unique to ddpg (not in vpg)
                 'update_every': 50,        # unique to ddpg (not in vpg)
                 'act_noise': 0.1,          # unique to ddpg (not in vpg)
                 'num_test_episodes': 10}   # unique to ddpg (not in vpg)
           }

IMPLEMENTED_ALGOS = PRESETS.keys()

# provides a default ActorCritic
DEFAULT_ACTOR_CRITIC = {"vpg": "MLPActorCritic",
                        "ddpg": "MLPActorCritic"}


# how the experiment directories are written
def outdir_from_preset(algo, use_custom, exp):
    if use_custom:
        variant = 'HLML'
    else:
        variant = 'spinningup'
    # e.g. 'vpg_HLML_LunarLander-v2' or 'vpg_spinningup_LunarLander-v2'
    experiment_name = '%s_%s_%s' % (algo, variant, exp)
    return experiment_name
