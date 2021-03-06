# AI_Gym
A collection of reinforcement learning approaches to games from the OpenAI gym

https://gym.openai.com/

We have written some convenience wrappers around the source code developed by
OpenAI, to make it easier for the user to play around with network architectures
in the context of a time-limited hackathon.

The `HLML_RL` class is defined in `rl_class.py` and is the main object used for
writing quick scripts such as `run.py` with custom PyTorch models. To build a
custom model, the user must write their own ActorCritic in
customActorCritic.py, following the documentation provided by the
`HLML_RL.ac_help()` method.

**Minimal user steps for training your own agent:**
1) Choose an OpenAI RL environment: https://github.com/openai/gym/wiki/Table-of-environments. Default: `'LunarLander-v2'`.
2) Choose a compatible RL training algorithm (see `compatibility_checks.py`). Default: `'vpg'`. 
3) In `customActorCritic.py`, specify your chosen algorithm as `TRAINING_ALG` 
4) In `customActorCritic.py`, specify your default algorithm hyperparameters in `CUSTOM_AC_DEFAULT_KWARGS`
5) In `customActorCritic.py`, write an ActorCritic following the documentation provided by
  the `HLML_RL.ac_help()` method.
6) Train the model using `run.py` as described below. 

**Training the model:**
- Modify `user_input` in `run.py`, which is used to initialize an instance as `HLML_RL(**user_input)`
- Modify  `train_input` in `run.py`, which is used to set hyperparameters for training in `HLML_RL.train(**train_input)`
- Run `run.py`

**Hardware Case 1: CPU only, multiple threads**
- Modify the default parameters for the training algorithm being used
- Specifically, need to override `ncpu: 1` in `user_input` defined in `run.py`.
- Only some algorithms (e.g. VPG) support MPI for multi-processing (use `ncpu=1` otherwise).

**Hardware Case 1: GPU**
- TODO
