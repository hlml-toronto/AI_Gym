import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from skimage.color import rgb2gray
import skimage.transform as transform

#import algos.vpg.core as core #TODO needs to change
from rl_class import get_IO_dim

# ------------------------------------------------------------------------------
# This file is where a custom ActorCritic can be defined by the user to be used
# in RL training in run.py
# ------------------------------------------------------------------------------

# specify the name of the training algorithm for your custom ActorCritic
TRAINING_ALG = 'vpg'

# write default hyperparameters for your custom ActorCritic
CUSTOM_AC_DEFAULT_KWARGS = { "resize_dim" : (84,84,1)
                            , "hidden_sizes" : (32,64,64)
                            , "hidden_kernel" : [8,4,2]
                            , "hidden_stride" : [4,2,1]
                            , "hidden_padding" : [0,0,0]
                            }


class customActor(nn.Module): # Was core.Actor
    """
    Custom user policy model
    """
    def __init__(self, resize_dim, act_dim, hidden_sizes, hidden_kernel
                    , hidden_stride, hidden_padding):
        super().__init__()
        # TODO make have kwargs pass filtering and such
        sizes = [resize_dim[2]] + list(hidden_sizes)
        layers = []
        new_img_size = resize_dim[0] # size of image after convolution
        for j in range(len(sizes)-1):
            # convolutional layer
            layers += [ nn.Conv2d(sizes[j], sizes[j+1]
                        , kernel_size = hidden_kernel[j]
                        , stride = hidden_stride[j]
                        , padding = hidden_padding[j])]
            #layers += [ nn.BatchNorm2d(sizes[j+1])]
            layers += [ nn.ReLU(inplace=True) ]
            #layers += [ nn.MaxPool2d(kernel_size=2, stride=2) ]

            # check that the new image size is an integer
            new_img_size = 1 + ( new_img_size - hidden_kernel[j]
                            - 2 * hidden_padding[j] ) / hidden_stride[j]
            print(new_img_size)
            try:
                assert new_img_size.is_integer()
            except AssertionError:
                sys.exit("Stride, padding and kernel cannot tile the input. " +
                            "Try different values.")

        # dropout layer (randomly sets certain input to 0) to avoid overfitting
        # (probably not necessary here)
        layers += [ nn.Dropout() ]

        # fully connected layer to activation layer
        layers += [ nn.Linear( int(new_img_size * new_img_size * sizes[-1]), act_dim) ]
        self.logits_net = nn.Sequential(*layers)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


# TODO determine if these are VPG specific; move if so
class customCritic(nn.Module):
    """
    Custom user on-policy value function model
    """
    def __init__(self, resize_dim, hidden_sizes, hidden_kernel, hidden_stride
                                , hidden_padding):
        super().__init__()
        sizes = [resize_dim[2]] + list(hidden_sizes)
        layers = []
        new_img_size = resize_dim[0] # size of image after convolution
        for j in range(len(sizes)-1):
            # convolutional layer
            layers += [ nn.Conv2d(sizes[j], sizes[j+1]
                        , kernel_size = hidden_kernel[j]
                        , stride = hidden_stride[j]
                        , padding = hidden_padding[j])]
            #layers += [ nn.BatchNorm2d(sizes[j+1])]
            layers += [ nn.ReLU(inplace=True) ]
            #layers += [ nn.MaxPool2d(kernel_size=2, stride=2) ]

            new_img_size = 1 + ( new_img_size - hidden_kernel[j]
                            - 2 * hidden_padding[j] ) / hidden_stride[j]

            try:
                assert new_img_size.is_integer()
            except AsserionError:
                sys.exit("Stride, padding and kernel cannot tile the input. " +
                            "Try different values.")

        # dropout layer (randomly sets certain input to 0) to avoid overfitting
        # (probably not necessary here)
        layers += [ nn.Dropout() ]

        # fully connected layer to activation layer, which is 1 number
        layers += [ nn.Linear( int(new_img_size * new_img_size * sizes[-1]), 1) ]
        self.v_net = nn.Sequential(*layers)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)  # Critical to ensure v has right shape.


class customActorCritic(nn.Module):
    """Sample code for a custom ActorCritic class to be used with the VPG
    training algorithm.
    """
    def __init__(self, obs_env, act_env, resize_dim=(64, 64, 1)
                    , hidden_sizes=[8,8,8] , hidden_kernel=[3,3,3]
                    , hidden_stride=[1,1,1], hidden_padding=[0,0,0]):
        super().__init__()
        obs_dim, act_dim = get_IO_dim((obs_env, act_env))
        # policy builder depends on action space
        self.resize_dim = resize_dim

        self.pi = customActor(resize_dim, act_dim, hidden_sizes, hidden_kernel
                                        , hidden_stride, hidden_padding)

        # build value function
        self.v = customCritic(resize_dim, hidden_sizes, hidden_kernel
                                        , hidden_stride, hidden_padding )

    def transform_obs(self, obs):
        # Need to transform data to make the network less large.
        # Here, we've simply made the image greyscale and then downscaled it
        grey_obs = rgb2gray(obs)
        return transform.resize(grey_obs, self.resize_dim)

    def step(self, obs):
        resize_obs = self.transform_obs( obs )
        with torch.no_grad():
            pi = self.pi._distribution(resize_obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(resize_obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]
