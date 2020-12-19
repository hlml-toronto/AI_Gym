# choose if MPI multi-threading (n > 1) or not (n = 1)
DEFAULT_NCPU = 1

# each algorithm has at least one or two variants
DEFAULT_VARIANT = 'spinningup'
assert DEFAULT_VARIANT in ['spinningup', 'HLML']

# prepare the default environment
DEFAULT_ENV_STR = 'LunarLander-v2'
TESTED_ENVS = ['LunarLander-v2']

# define default parameters for each implemented training algorithm (and each variant)
# TODO Large suggestion: we make separate 'custom' scripts for each algo,
#  hlml_vpg.py, hlml_ddpg.py, ...
#  the users construct their own nn's in these scripts (we provide our own simple templates)
#  define the ac and w.e ac_kwargs are needed/manipulated there

# TODO then we package everything centrally here as follows
#  PRESETS['ddpg']['spinningup']['ac'] = getattr(core, "MLPActorCritic")
#  PRESETS['ddpg']['spinningup']['ac_kwargs'] = dict(hidden_sizes=[64] * 2)
#  ...
#  PRESETS['ddpg']['HLML']['ac'] = HLML_ActorCritic
#  PRESETS['ddpg']['HLML']['ac_kwargs'] = {'pi': myActor, 'pi': myCritic}  (we can also rethink how ac_kwargs are used)

# TODO supposing we do any of the above,
#  this presets object might be better as a class,
#  including a method for saving args to file
# TODO currently the big internal dict of kwargs can only contain kwargs for spinningup vpg, ddpg, etc (how to generalize)?

PRESETS = {'vpg':
               {'spinningup':
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
                'HLML':
                    {'gamma': 0.99,
                     'seed': 0,
                     'steps_per_epoch': 4000,
                     'epochs': 50,
                     'pi_lr': 3e-4,
                     'vf_lr': 1e-3,
                     'max_ep_len': 1000,
                     'save_freq': 10,
                     'logger_kwargs': dict(),
                     'train_v_iters': 80,
                     'lam': 0.97},
                },
           'ddpg':
               {'spinningup':
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
                'HLML':
                    {'gamma': 0.99,
                     'seed': 0,
                     'steps_per_epoch': 4000,
                     'epochs': 100,
                     'pi_lr': 1e-3,
                     'q_lr': 1e-3,
                     'max_ep_len': 1000,
                     'save_freq': 1,
                     'logger_kwargs': dict(),
                     'replay_size': int(1e6),
                     'polyak': 0.995,
                     'batch_size': 100,
                     'start_steps': 10000,
                     'update_after': 1000,
                     'update_every': 50,
                     'act_noise': 0.1,
                     'num_test_episodes': 10},
                },
           }

IMPLEMENTED_ALGOS = PRESETS.keys()


# how the experiment directories are written
def outdir_from_preset(algo, variant):
    experiment_name = '%s_%s' % (algo, variant)  # e.g. 'vpg_HLML' or 'vpg_spinningup'
    return experiment_name
