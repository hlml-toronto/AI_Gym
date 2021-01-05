import gym, torch, os, re
import matplotlib.pyplot as plt
import torch.nn as nn
from gym.spaces import Box, Discrete
from importlib import import_module
from matplotlib import animation
import copy

import utils.test_policy as test_policy
from presets import PRESETS, IMPLEMENTED_ALGOS, TESTED_ENVS, DEFAULT_ACTOR_CRITIC
from utils.mpi_tools import mpi_fork
from utils.run_utils import setup_logger_kwargs
from compatibility_checks import COMPATIBILITY_CHECKS


def get_IO_dim(arg):
    """ Given an environment name or a tuple of (observation space, action space),
    return obs_dim, act_dim representing the dimension of the state vector and
    the dimension of the action vector respectively.
    """
    if type(arg) == str:  # input is AI Gym environment name
        test_env = gym.make(arg)
        observation_space = test_env.observation_space
        action_space = test_env.action_space
    elif type(arg) == tuple:  # input is (observation space, action space)
        observation_space, action_space = arg
    else:
        raise TypeError("Did not recognize type of input to get_IO_dim()")
    # get obj_dim
    if isinstance(observation_space, Box):
        obs_dim = observation_space.shape[0]
    else:
        assert isinstance(observation_space, Discrete)
        obs_dim = observation_space.n
    # get act_dim
    if isinstance(action_space, Box):
        act_dim = action_space.shape[0]
    else:
        assert isinstance(action_space, Discrete)
        act_dim = action_space.n

    return obs_dim, act_dim


class HLML_RL:
    """
    Base class for easy reinforcement learning with OpenAI Gym.

    ...

    Attributes
    ----------
    training_alg : str
        A keyword string indicating which training algorithm to use
    use_custom : Bool
        A Boolean indicating whether or not to use a custom variant of the
        ActorCritic during training
    actorCritic : nn.Module
        (default value is None)
        An instance of a PyTorch module implementing an actor and a critic,
        as specified by the training algorithm.

        If None is passed then default arguments from core.MLPActorCritic will
        be used, and default PyTorch models will be generated. If the default
        architectures are desired but with different hidden layer sizes,
        keword arguments can be passed through ac_kwargs keword in **kwargs.
    env : str
        The OpenAI Gym environment name to instantiate and train the RL agent on

    Methods
    -------
    ac_help()
        Prints documentation for the ActorCritic class needed given the
        training algorithm chosen, to help the user build a custom ActorCritic
    train()
        Trains the RL agent on the environment `env` according to the algorithm
        specified by `training_alg`, using the models provided in `model_list`
    render()
        Generates a short movie of the RL agent acting in the environment
        according to its current parameterization
    load()
        Load a previously trained RL model
    save()
        Store the current parameterization of the RL agent for future use
    """
    def __init__(self, **kwargs):
        self.training_alg = kwargs['training_alg']
        self.use_custom = kwargs['use_custom']
        self.actorCritic = kwargs['actorCritic']
        self.env_str = kwargs['env_str']
        self.env = lambda: gym.make(self.env_str)
        self.ncpu = kwargs['ncpu']
        self.exp_name = kwargs['exp_name']
        self.seed = kwargs.get('seed', 0)

        # check algorithm is implemented and environment has been tested
        assert self.training_alg in IMPLEMENTED_ALGOS
        if self.env_str not in TESTED_ENVS:
            print("The current environment has not been tested. Run at your own risk!")

        # algo versus environment compatibility check
        #  e.g. spinningup ddpg assumes continuous action space, lunar lander is discrete though

        if self.training_alg in COMPATIBILITY_CHECKS.keys():
            test_env = gym.make(self.env_str)
            observation_space = test_env.observation_space
            action_space = test_env.action_space
            try:
                assert type(observation_space) in COMPATIBILITY_CHECKS[self.training_alg]['obs_env']
                assert type(action_space) in COMPATIBILITY_CHECKS[self.training_alg]['act_env']
            except AssertionError:
                raise AssertionError("\n\n\nThe gym environment and training algorithm selected are not compatible! Please check what type of action space and state space are required by your training algorithm, or try with a different gym environment.")
            # ncpu warnings
            if self.ncpu > 1 and COMPATIBILITY_CHECKS[self.training_alg]['ncpu_warn']:
                print('Warning: ncpu > 1 (set to %d) is unstable with %s' % (self.ncpu, self.training_alg))
        else:
            print("The current training algorithm does not have any listed compatible environments. Run at your own risk!")

    def ac_help(self):
        """Prints the documentation for the ActorCritic class specific to the
        user's selected training algorithm.
        """
        # dynamically import source code (e.g. import algos.vpg.vpg as mod)
        mod = import_module("algos.{}.{}".format(self.training_alg, self.training_alg))
        method = getattr(mod, self.training_alg)  # e.g. from algos.vpg.vpg import vpg
        docstring = method.__doc__
        actor_critic_doc = re.search('actor_critic:(.*)ac_kwargs',
                                     docstring, re.DOTALL).group(1)
        head = self.training_alg.upper() + "_ActorCritic\n\t\"\"\"\n\t"
        foot = "\"\"\""
        actor_critic_doc = head + actor_critic_doc + foot
        print(actor_critic_doc)

    def train(self, **kwargs):
        """
        Run the training algorithm to optimize model parameters for the
        environment provided.
        """
        # define default parameters for each training algorithm, then perturb them based on user input
        preset_kwargs = PRESETS[self.training_alg]   # select default kwargs for the algo
        if self.use_custom:  # do not pass default ac_kwargs if custom ActorCritic is specified
            preset_kwargs.pop('ac_kwargs')
        preset_kwargs.update(kwargs)                 # update default algo kwargs based on user input

        # dynamically import source code (e.g. import algos.vpg.vpg as mod)
        mod = import_module("algos.{}.{}".format(self.training_alg, self.training_alg))
        method = getattr(mod, self.training_alg)  # e.g. from algos.vpg.vpg import vpg

        if self.actorCritic is None:
            # use the default actorCritic for the algo
            core = import_module("algos.{}.core".format(self.training_alg))  # e.g. import algos.vpg.core as core
            self.actorCritic = getattr(core, DEFAULT_ACTOR_CRITIC[self.training_alg])  # e.g. from core import MLPActorCritic as actorCritic

        # prepare mpi if self.ncpu > 1 (and supported by chosen RL algorithm)
        mpi_fork(self.ncpu)  # run parallel code with mpi

        # update logger kwargs
        logger_kwargs = setup_logger_kwargs(self.exp_name, preset_kwargs['seed'])
        preset_kwargs['logger_kwargs'] = logger_kwargs

        # begin training
        if kwargs['render_freq'] is None:
            method(self.env, actor_critic=self.actorCritic, **preset_kwargs)
        else:
            train_shedule = [kwargs['render_freq'] for _ in range(int(preset_kwargs['epochs'] / kwargs['render_freq']))] + [preset_kwargs['epochs'] % kwargs['render_freq']]
            train_kwargs_copy = copy.deepcopy(preset_kwargs)
            train_kwargs_copy.pop('render_freq')
            for idx, i in enumerate(train_shedule):
                train_kwargs_copy['epochs'] = i
                method(self.env, actor_critic=self.actorCritic, **train_kwargs_copy)
                render_kwarg = {'filename': '/gym_animation_' + str(idx + 1) + '.gif'}
                self.render(save=True, show=False, seed=self.seed, **render_kwarg)


    def load_agent(self, seed=0):
        """Load to pick up training where left off
        """
        pytsave_path = os.path.join("experiments",
                                    self.exp_name,
                                    self.exp_name + "_s" + str(seed),
                                    'pyt_save')
        self.ac = torch.load(os.path.join(pytsave_path, "model.pt"))
        return pytsave_path

    def render(self, seed=0, save=False, show=True, *args, **kwargs):
        # logger_kwargs = {'output_dir' : "Jeremy", "exp_name" : whichever}
        save_path = self.load_agent(seed)

        if show:
            render = True; max_ep_len = None; num_episodes = 20; itr = 'last'
            deterministic = 100
            env, get_action = test_policy.load_policy_and_env(save_path, itr, deterministic)
            test_policy.run_policy(env, get_action, max_ep_len, num_episodes, render)
        if save:
            """
            Code from botforge:
                https://gist.github.com/botforge/64cbb71780e6208172bbf03cd9293553
            Ensure you have imagemagick installed with
            sudo apt-get install imagemagick
            """
            fname = kwargs.get('filename', '/gym_animation.gif')

            def save_frames_as_gif(frames, path=save_path, filename=fname):
                # Mess with this to change frame size
                plt.figure(figsize=(frames[0].shape[1] / 72.0,
                                    frames[0].shape[0] / 72.0), dpi=72)
                patch = plt.imshow(frames[0]); plt.axis('off')

                def animate(i):
                    patch.set_data(frames[i])

                anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
                anim.save(path + filename, writer='imagemagick', fps=60)

            # Make gym env
            env = gym.make(self.env_str)

            # Run the env
            obs = env.reset(); frames = []
            for t in range(1000):
                # Render to frames buffer
                frames.append(env.render(mode="rgb_array"))
                action = self.ac.act(torch.as_tensor(obs, dtype=torch.float32))
                obs, res, done, _ = env.step(action)
                if done:
                    break
            env.close()
            save_frames_as_gif(frames)

        return 0
