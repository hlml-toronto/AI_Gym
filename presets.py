# choose if MPI multi-threading (n > 1) or not (n = 1)
DEFAULT_NCPU = 1

# prepare the default environment
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
                 'num_test_episodes': 10},  # unique to ddpg (not in vpg)

           'sac':
                {'ac_kwargs': dict(),
                 'seed': 0,
                 'steps_per_epoch': 4000,
                 'epochs': 100,
                 'replay_size': 1000000,
                 'gamma': 0.99,
                 'polyak': 0.995,
                 'lr': 0.001,
                 'alpha': 0.2,
                 'batch_size': 100,
                 'start_steps': 10000,
                 'update_after': 1000,
                 'update_every': 50,
                 'num_test_episodes': 10,
                 'max_ep_len': 1000,
                 'logger_kwargs': dict(),
                 'save_freq': 1},

           'ppo':
                {'ac_kwargs': dict(),
                 'seed': 0,
                 'steps_per_epoch': 4000,
                 'epochs': 50,
                 'gamma': 0.99,
                 'clip_ratio': 0.2,
                 'pi_lr': 0.0003,
                 'vf_lr': 0.001,
                 'train_pi_iters': 80,
                 'train_v_iters': 80,
                 'lam': 0.97,
                 'max_ep_len': 1000,
                 'target_kl': 0.01,
                 'logger_kwargs': dict(),
                 'save_freq': 10},

           'td3':
                {'ac_kwargs': dict(),
                 'seed': 0,
                 'steps_per_epoch': 4000,
                 'epochs': 100,
                 'replay_size': 1000000,
                 'gamma': 0.99,
                 'polyak': 0.995,
                 'pi_lr': 0.001,
                 'q_lr': 0.001,
                 'batch_size': 100,
                 'start_steps': 10000,
                 'update_after': 1000,
                 'update_every': 50,
                 'act_noise': 0.1,
                 'target_noise': 0.2,
                 'noise_clip': 0.5,
                 'policy_delay': 2,
                 'num_test_episodes': 10,
                 'max_ep_len': 1000,
                 'logger_kwargs': dict(),
                 'save_freq': 1}
           }

IMPLEMENTED_ALGOS = PRESETS.keys()

# provides a default ActorCritic
DEFAULT_ACTOR_CRITIC = {"vpg": "MLPActorCritic",
                        "ddpg": "MLPActorCritic",
                        "sac": "MLPActorCritic",
                        "ppo": "MLPActorCritic",
                        "td3": "MLPActorCritic"}


# how the experiment directories are written
def outdir_from_preset(algo, use_custom, exp):
    if use_custom:
        variant = 'HLML'
    else:
        variant = 'spinningup'
    # e.g. 'vpg_HLML_LunarLander-v2' or 'vpg_spinningup_LunarLander-v2'
    experiment_name = '%s_%s_%s' % (algo, variant, exp)
    return experiment_name
