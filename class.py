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

    def train(self, *args, **kwargs):
        raise NotImplementedError

    def render(self, *args, **kwargs):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError
