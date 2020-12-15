from gym.spaces import Box, Discrete
import gym


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

    def render(self, *args, **kwargs):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError
