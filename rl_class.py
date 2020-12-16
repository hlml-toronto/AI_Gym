from gym.spaces import Box, Discrete
import gym
from importlib import import_module
import torch
import torch.nn as nn


class HLML_ActorCritic(nn.Module):
    """
    Must take initialization arguments: observation_space, action_space, **ac_kwargs
    """
    def __init__(self, observation_space, action_space, **ac_kwargs):
        super().__init__()
        if ac_kwargs['training_alg'] == 'vpg':
            self.pi = ac_kwargs['model_list'][0]
            self.v = ac_kwargs['model_list'][1]
        else:
            raise NotImplementedError("The training algorithm requested has not been implemented")

        # Check that user models are compatible with algorithm
        core = import_module("algos.{}.core".format(ac_kwargs['training_alg']))
        assert issubclass(type(self.pi), core.Actor)  # pi should have Actor methods
        assert issubclass(type(self.v), core.MLPCritic)  # v should have MLPCritic methods

        if isinstance(action_space, Box):
            test_act = self.pi._distribution(observation_space.sample())
            true_act = action_space.sample()
            if len(test_act) != len(true_act):
                raise TypeError("\nThe output of pi is of length {} but a vector of length {} was expected\n".format(test_act, true_act))
        elif isinstance(action_space, Discrete):
            test_act = self.pi._distribution(torch.from_numpy(observation_space.sample()))
            if isinstance(test_act, int):
                raise TypeError("\nThe expected output of pi is a discrete action represented by an integer\n")

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]


class HLML_RL:
    """
    Base class for easy reinforcement learning with OpenAI Gym.

    ...

    Attributes
    ----------
    training_alg : str
        A keyword string indicating which training algorithm to use
    model_list : list
        (default value is None)
        A list of PyTorch models needed to run the indicated training algorithm.

        If None is passed then default arguments from core.MLPActorCritic will
        be used, and default PyTorch models will be generated. If the default
        architectures are desired but with different hidden layer sizes,
        keword arguments can be passed through ac_kwargs keword in **kwargs.
    env : str
        The OpenAI Gym environment name to instantiate and train the RL agent on

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
        if self.env not in ['LunarLander-v2']:
            print("The current environment has not been tested. Run at your own risk!")

    def train(self, *args, **kwargs):
        """
        Run the training algorithm to optimize model parameters for the
        environment provided.
        """
        # Define default parameters for each training algorithm
        default_kwargs = {'vpg': {'env': lambda: gym.make(self.env),
                                  'ac_kwargs': dict(hidden_sizes=[64]*2),
                                  'gamma': 0.99,
                                  'seed': 0,
                                  'cpu': 1,
                                  'steps': 4000,
                                  'epochs': 50,
                                  'exp_name': 'vpg'}
                          }
        IO = default_kwargs[self.training_alg]  # Select appropriate kwargs
        IO.update(kwargs)  # update with user input

        # Dynamically import source code
        mod = import_module("algos.{}.{}".format(self.training_alg, self.training_alg))
        method = getattr(mod, self.training_alg)
        if self.model_list is None:
            core = import_module("algos.{}.core".format(self.training_alg))
            actorCritic = getattr(core, "MLPActorCritic")
        else:
            actorCritic = HLML_ActorCritic
            IO['ac_kwargs'] = {'model_list': self.model_list, 'training_alg': self.training_alg}

        from utils.mpi_tools import mpi_fork

        mpi_fork(IO['cpu'])  # run parallel code with mpi

        from utils.run_utils import setup_logger_kwargs
        logger_kwargs = setup_logger_kwargs(IO['exp_name'], IO['seed'])

        method(IO['env'], actor_critic=actorCritic, ac_kwargs=IO['ac_kwargs'],
               gamma=IO['gamma'], seed=IO['seed'], steps_per_epoch=IO['steps'],
               epochs=IO['epochs'], logger_kwargs=logger_kwargs)

    def render(self, *args, **kwargs):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError
