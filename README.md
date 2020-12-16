# AI_Gym
A collection of reinforcement learning approaches to games from the OpenAI gym

https://gym.openai.com/

We have written some convenience wrappers around the source code developed by
OpenAI, to make it easier for the user to play around with network architectures
in the context of a time-limited hackathon.

The `HLML_RL` class is defined in rl_class.py and is the main object used for
writing quick scripts such as run.py with custom PyTorch models. To build a
custom model, the user must write their own Actor and Critic classes, following
the class templates defined in algos.\<training algorithm\>.core .

To write a custom Actor (i.e. a model for the policy distribution pi) the
user should implement a new class which inherits from the Actor class in `core`
and has the methods *_distribution*, *_log_prob_from_distribution*, and
*forward*.

To write a custom Critic (i.e. for Vanilla Policy Gradient this is the
on-policy value function) the user should implement a new class which has a
*forward* method that returns the predicted value (as computed by V) for each
item in the input batch.
