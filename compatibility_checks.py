from gym.spaces import Box, Discrete
# ------------------------------------------------------------------------------
# This file is where compatibility between various components is encoded, to
# use for checks before running primary RL tasks
# ------------------------------------------------------------------------------

# TODO ncpu support (and maybe GPU) for different algorithms here

COMPATIBILITY_CHECKS = {'ddpg':
                        {'obs_env': [Box, Discrete],
                         'act_env': [Box]},

                        'vpg':
                        {'obs_env': [Box, Discrete],
                         'act_env': [Box, Discrete]},

                        'sac':
                        {'obs_env': [Box, Discrete],
                         'act_env': [Box]},

                        }
