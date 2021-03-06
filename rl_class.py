import gym, torch, os, re, shutil
import matplotlib.pyplot as plt
import torch.nn as nn
from gym.spaces import Box, Discrete
from importlib import import_module
from matplotlib import animation
import glob

import utils.test_policy as test_policy
from presets import PRESETS, IMPLEMENTED_ALGOS, TESTED_ENVS, DEFAULT_ACTOR_CRITIC
from utils.mpi_tools import mpi_fork
from utils.run_utils import setup_logger_kwargs
from compatibility_checks import COMPATIBILITY_CHECKS


def get_IO_dim(arg):
    """ Given an environment name or a tuple of (observation space, action space),
    return obs_dim, act_dim as tuples representing the dimension of the state
    vector and the dimension of the action vector respectively.
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
        obs_dim = observation_space.shape   #TODO [0] bug?
    else:
        assert isinstance(observation_space, Discrete)
        obs_dim = (observation_space.n,)
    # get act_dim
    if isinstance(action_space, Box):
        act_dim = action_space.shape        #TODO [0] bug?
    else:
        assert isinstance(action_space, Discrete)
        act_dim = (action_space.n,)

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
    restart_tuple : two-tuple; None
        The exp_name and exp_name_s%d of the original run which is being restarted

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
        self.restart_tuple = kwargs.get('restart_tuple', None)

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
        preset_kwargs.update(kwargs)                 # update default algo kwargs based on user input
        render_saves = preset_kwargs.get('render_saves', False)
        if 'render_saves' in preset_kwargs.keys():
            preset_kwargs.pop('render_saves')

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
        method(self.env, actor_critic=self.actorCritic, **preset_kwargs)

        # render all checkpoints user specifies with 'render_saves'
        if render_saves:
            log_dir = logger_kwargs['output_dir'] + os.sep + 'pyt_save' + os.sep
            fnames = glob.glob(log_dir + 'model*.pt')[1:]  # first item in list is final checkpoint, with no itr in file name
            for checkpoint in fnames:
                itr = re.search('model(.*).pt', checkpoint).group(1) # get epoch number from file name
                render_kwargs = {'filename': '/gym_animation_' + str(itr) + '.mp4',
                                 'model_itr': itr}
                self.render(save=True, show=False, seed=self.seed, **render_kwargs)
            #self.render(save=True, show=False, seed=self.seed)  # also render the final trained model


    def restart_train(self, **kwargs):
        """
        Restarts training based on the the original run specified by self.restart_tuple
        """
        # extract original run info
        assert self.restart_tuple is not None
        orig_exp_name = self.restart_tuple[0]
        orig_exp_name_with_seed = self.restart_tuple[1]
        orig_exp_seed = orig_exp_name_with_seed.split('s')[-1]
        orig_exp_pytdir = 'experiments' + os.sep + orig_exp_name + os.sep + orig_exp_name_with_seed + os.sep + 'pyt_save'
        orig_exp_models = sorted([a for a in os.listdir(orig_exp_pytdir) if a[-3:] == '.pt'])
        orig_exp_lastmodel = orig_exp_models[-1]                          # last file will be the one with last epoch
        orig_exp_epochs = int(orig_exp_lastmodel[:-3].split('l')[1]) + 1  # e.g. "models99.pt" will give 100

        # load final model from original training
        # Note: this sets self.actorCritic, which is passed to method() in train()
        self.load_agent(seed=orig_exp_seed,
                        model_itr="%d" % (orig_exp_epochs - 1),
                        exp_name=orig_exp_name)

        # call train (will output to the working directory with 'restart' suffix)
        self.train(**kwargs)

        # rename and move files to original directory, modifying the filenames  # TODO
        # - model(%d).pt -> model(%d + epochs).pt
        # - gym_animation_(%d).pt -> gym_animation_(%d + epochs).pt
        # outside of pyt_save:
        # - vars(%d).pt -> vars(%d + epochs).pt
        # - 'progress.txt' -> 'progress_restart.txt'
        # - 'config.json' -> 'config_restart.json'
        working_dir = 'experiments' + os.sep + self.exp_name + os.sep + self.exp_name + '_s%d' % self.seed
        dest = 'experiments' + os.sep + orig_exp_name + os.sep + orig_exp_name_with_seed
        working_dir_pyt = working_dir + os.sep + 'pyt_save'
        dest_pyt = dest + os.sep + 'pyt_save'

        def batch_rename_and_move(front, ext, parent_dir, dest_dir):
            # TODO care off by 1 errors
            # acts like `ls front*ext` in command line
            numbered_files = sorted(glob.glob(parent_dir + os.sep + '%s*%s' % (front, ext)))[::-1]
            for fpath in numbered_files:
                print('numbered file fpath', fpath)
                itr = re.search('%s(.*)%s' % (front, ext), fpath).group(1)  # get epoch number from file name
                if len(itr) > 0:
                    new_epoch_num = orig_exp_epochs + int(itr)
                    new_fpath = re.sub('%s(.*)%s' % (front, ext),
                                       front + str(new_epoch_num) + ext, fpath)
                    os.rename(fpath, new_fpath)
                    print('renaming \n%s to \n%s' % (fpath, new_fpath))
                    fpath = new_fpath
                # move file to new directory (and overwrite)
                shutil.move(fpath, dest_dir + os.sep + os.path.basename(fpath))
            return

        batch_rename_and_move('model',          '.pt',  working_dir_pyt, dest_pyt)
        batch_rename_and_move('gym_animation_', '.gif', working_dir_pyt, dest_pyt)
        batch_rename_and_move('vars',           '.pkl', working_dir,     dest)

        # rename and move misc files
        # progress file
        os.rename(working_dir + os.sep + 'progress.txt',
                  working_dir + os.sep + 'progress_restart.txt')
        shutil.move(working_dir + os.sep + 'progress_restart.txt',
                    dest + os.sep + 'progress_restart.txt')
        # config file
        os.rename(working_dir + os.sep + 'config.json',
                  working_dir + os.sep + 'config_restart.json')
        shutil.move(working_dir + os.sep + 'config_restart.json',
                    dest + os.sep + 'config_restart.json')

        # delete the working directory (the restart directory)
        shutil.rmtree(os.path.join('experiments', self.exp_name))

        return

    def load_agent(self, seed=0, model_itr="", exp_name=None):
        """Load to pick up training where left off
        """
        if exp_name is None:
            exp_name = self.exp_name
        pytsave_path = os.path.join("experiments",
                                    exp_name,
                                    exp_name + "_s" + str(seed),
                                    'pyt_save')
        self.actorCritic = torch.load(os.path.join(pytsave_path, "model" + model_itr + ".pt"))
        return pytsave_path

    def render_video(self, seed=0, model_itr="", exp_name=None):
        self.load_agent(seed, model_itr, exp_name)
        # Make gym env
        env = gym.make(self.env_str)
        env = gym.wrappers.Monitor(env, './video/', force = True)

        # Run the env
        obs = env.reset(); frames = []
        for t in range(1000):
            env.render()
            action = self.actorCritic.act(torch.as_tensor(obs, dtype=torch.float32))
            obs, res, done, _ = env.step(action)
            if done:
                break
        env.close()

        return 0

    def render(self, seed=0, save=False, show=True, pytsave_path=None,
                        video_fmt='avi', *args, **kwargs):

        if pytsave_path is None:
            save_path = self.load_agent(seed, model_itr=kwargs.get('model_itr', ""))
        else:
            save_path = pytsave_path

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
            sudo apt-get install ffmpeg
            For some reason, imagemagick has some problems with fps, using
            ffmpeg instead
            """
            fname = kwargs.get('filename', '/gym_animation.' + video_fmt)

            def save_frames_as_gif(frames, path=save_path, filename=fname):
                # Mess with this to change frame size
                plt.figure(figsize=(frames[0].shape[1] / 72.0,
                                    frames[0].shape[0] / 72.0), dpi=72)
                patch = plt.imshow(frames[0]); plt.axis('off')

                def animate(i):
                    patch.set_data(frames[i])

                anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
                anim.save(path + filename, writer='ffmpeg', fps=60 )

            # Make gym env
            env = gym.make(self.env_str)

            # Run the env
            obs = env.reset(); frames = []
            for t in range(1000):
                # Render to frames buffer
                frames.append(env.render(mode="rgb_array"))
                action = self.actorCritic.act(torch.as_tensor(obs, dtype=torch.float32))
                obs, res, done, _ = env.step(action)
                if done:
                    break
            env.close()
            save_frames_as_gif(frames)

        return 0
