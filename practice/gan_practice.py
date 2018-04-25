# other utils
import numpy as np
import multiprocessing as mp
import sys

# torch autograd, nn, etc.
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DistributionGenerator(nn.Module):

    def __init__(self, input_noise_distribution, input_dim, output_dim, hidden_dim):
        super(DistributionGenerator, self).__init__()
        self.input_noise_distribution = input_noise_distribution
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        # TODO: play with these; note this is just a regular feedforward net
        # (no convolutions, pooling, etc.)
        self.fc1_nonlin = F.elu
        self.fc2_nonlin = F.relu
        self.fc1_lin = nn.Linear(input_dim, hidden_dim)
        self.fc2_lin = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_lin = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.f1_nonlin(self.fc1_lin(x))
        x = self.f2_nonlin(self.fc2_lin(x))
        x = self.f3_lin(x)
        return x

    def generate_sample(self):
        x = self.input_noise_distribution(input_size)
        return self.forward(x)

class DistributionDiscriminator(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(DistributionDiscriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # TODO: play with these; note this is just a regular feedforward net
        # (no convolutions, pooling, etc.)
        self.fc1_nonlin = F.elu
        self.fc2_nonlin = F.relu
        self.fc1_lin = nn.Linear(input_dim, hidden_dim)
        self.fc2_lin = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_lin = nn.Linear(hidden_dim, 2)
        # NOTE: this isn't strictly necessary because one can just take a max over the
        # output of the final linear layer; but it's nice to have a probability distribution
        self.make_prob = nn.Softmax 

    def forward(self, x):
        x = self.f1_nonlin(self.fc1_lin(x))
        x = self.f2_nonlin(self.fc2_lin(x))
        x = self.make_prob(self.f3_lin(x))
        return x

    def discriminate_sample(self, candidate_sample):
        probs = self.forward(candidate_sample)
        return probs.argmax()

class DistributionGAN:

    def __init__(self,
                 input_noise_distribution,
                 target_distribution,
                 noise_dim,
                 sample_dim,
                 gen_hidden_dim,
                 dis_hidden_dim):
        self.input_noise_distribution = input_noise_distribution
        self.target_distribution = target_distribution
        self.generator = DistributionGenerator(input_noise_distribution, noise_dim, sample_dim, gen_hidden_dim)
        self.discriminator = DistributionDiscriminator(sample_dim, dis_hidden_dim)

    # train using vanilla Jensen-Shannon divergence/KL divergence (see vanilla GAN)
    def jensen_shannon_train(self):
        pass # TODO

    # train using wasserstein L1 distance (see WGAN)
    def wasserstein_train(self):
        pass # TODO

    # train using least squares divergence (see LSGAN)
    def least_squares_train(self):
        pass # TODO

if __name__ == "__main__":
    input_noise_distribution = lambda size : np.random.normal(loc=0.0, scale=1.0, size=size)
    target_distribution = lambda size : np.random.chisquare(df=10, size=size)

    noise_dim = 10 # dimensionality of noise distribution
    sample_dim = 10 # dimensionality of generator output (and target distribution)
    gen_hidden_dim = 20 # hidden layer size for generator
    dis_hidden_dim = 20 # hidden layer size for discriminator

    gan = DistributionGAN(input_noise_distribution, target_distribution,
                          noise_dim, sample_dim, gen_hidden_dim, dis_hidden_dim)

