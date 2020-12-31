from gym.spaces import Box, Discrete
# ------------------------------------------------------------------------------
# This file is where compatibility between various components is encoded, to
# use for checks before running primary RL tasks
# ------------------------------------------------------------------------------

COMPATIBILITY_CHECKS = {'ddpg':
                        {'spinningup':
                            {'obs_env': [Box, Discrete],
                             'act_env': [Box]},
                         'HLML':
                            {'obs_env': [Box, Discrete],
                             'act_env': [Box]},
                         },

                        'vpg':
                        {'spinningup':
                            {'obs_env': [Box, Discrete],
                             'act_env': [Box, Discrete]},
                         'HLML':
                            {'obs_env': [Box, Discrete],
                             'act_env': [Box, Discrete]},
                         }
                        }
