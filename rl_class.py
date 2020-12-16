from gym.spaces import Box, Discrete
import gym

from matplotlib import animation
import matplotlib.pyplot as plt
import utils.test_policy as test_policy

class HLML_RL:
    """
    Base class for easy reinforcement learning with OpenAI Gym.

    ...

    Attributes
    ----------
    training_alg : str
        a keyword string indicating which training algorithm to use
    model_list : list
        a list of PyTorch models needed to run the indicated training algorithm
    env : Env
        the OpenAI Gym environment to train the RL agent on

    Methods
    -------
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
    def __init__(self, *args, **kwargs):
        self.training_alg = kwargs['training_alg']
        self.model_list = kwargs['model_list']
        self.env = kwargs['env']
        # Check that requested training algorithm is implemented
        assert self.training_alg in ['vpg']
        # Check that the environment has been tested
        if self.env.spec.id not in ['LunarLander-v2']:
            print("The current environment has not been tested. Run at your own risk!")
        # Check that the models provided are compatible with the environment
        self.observation_space_size = len(self.env.observation_space.sample())
        if isinstance(self.env.action_space, Discrete):
            self.action_space_size = 1
        elif isinstance(self.env.action_space, Box):
            self.action_space_size = len(self.env.action_space.sample())

        if self.training_alg == 'vpg':
            pass
            # TO DO: check that models provided output correct size vectors
            # Check that V input matches action_space_size and observation_space_size
            # Check that pi input matches action_space_size

    def __VPG__(self, **kwargs):
        """
        Run vanilla policy gradient training algorithm by calling algo.vpg.vpg
        """
        from algos.vpg import vpg
        from algos.vpg import core
        from utils.mpi_tools import mpi_fork

        mpi_fork(kwargs['cpu'])  # run parallel code with mpi

        from utils.run_utils import setup_logger_kwargs
        logger_kwargs = setup_logger_kwargs(kwargs['exp_name'], kwargs['seed'])

        vpg.vpg(kwargs['env'], actor_critic=core.MLPActorCritic,
                ac_kwargs=dict(hidden_sizes=[kwargs['hid']]*kwargs['l']),
                gamma=kwargs['gamma'],
                seed=kwargs['seed'], steps_per_epoch=kwargs['steps'],
                epochs=kwargs['epochs'], logger_kwargs=logger_kwargs)

    def train(self, *args, **kwargs):
        """
        Run the training algorithm to optimize model parameters for the
        environment provided.
        """
        if self.training_alg == 'vpg':
            default_kwargs = {'env': lambda: gym.make('LunarLander-v2'),
                              'hid': 64,
                              'l': 2,
                              'gamma': 0.99,
                              'seed': 0,
                              'cpu': 1,
                              'steps': 4000,
                              'epochs': 50,
                              'exp_name': 'vpg'}
            default_kwargs.update(kwargs)
            self.__VPG__(**default_kwargs)


    # Load to pick up training where left off?
    def load_agent(self, save_path):
        self.exp_name = "XYZ.pt"
        self.ac = torch.load(save_path + os.sep + self.exp_name)
        return 0

    def render(self, save_path, save=False, show=True, *args, **kwargs):
        # logger_kwargs = {'output_dir' : "Jeremy", "exp_name" : whichever}
        load_agent(save_path)

        if show:
            env, get_action = load_policy_and_env(args.fpath
                                        , args.itr if args.itr >=0 else 'last'
                                        , args.deterministic)
            run_policy(env, get_action, args.len, args.episodes
                                        , not(args.norender))
        if save:
            """
            Code from botforge:
                https://gist.github.com/botforge/64cbb71780e6208172bbf03cd9293553
            Ensure you have imagemagick installed with
            sudo apt-get install imagemagick
            """
            def save_frames_as_gif(frames, path='./'
                                            , filename='gym_animation.gif'):

                #Mess with this to change frame size
                plt.figure(figsize=(frames[0].shape[1] / 72.0
                            , frames[0].shape[0] / 72.0), dpi=72)
                patch = plt.imshow(frames[0]); plt.axis('off')

                def animate(i):
                    patch.set_data(frames[i])

                anim = animation.FuncAnimation(plt.gcf(), animate
                                        , frames = len(frames), interval=50)
                anim.save(path + filename, writer='imagemagick', fps=60)

            #Make gym env
            env = gym.make('CartPole-v1')

            #Run the env
            observation = env.reset(); frames = []
            for t in range(1000):
                #Render to frames buffer
                frames.append(env.render(mode="rgb_array"))
                action = ac.act(torch.as_tensor(obs, dtype=torch.float32))
                o, r, d, _ = env.step(a)
                obs, res, done, _ = env.step(action)
                if done:
                    break
            env.close()
            save_frames_as_gif(frames)

    return 0

    # Super easy. Not sure why we'd want this (for restarting where we left off
    # training perhaps)


    # Done automatically.
    def save_agent(self, save_path):
        raise NotImplementedError
