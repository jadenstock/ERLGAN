# other utils
import numpy as np
#import multiprocessing as mp
import sys

# torch autograd, nn, etc.
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# plotting
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from scipy import stats


class GAN:

    def __init__(self, generator, discriminator):
        self.generator = generator
        self.discriminator = discriminator

    # train using vanilla Jensen-Shannon divergence/KL divergence (see vanilla GAN)
    # (see also https://github.com/devnag/pytorch-generative-adversarial-networks/blob/master/gan_pytorch.py)
    def jensen_shannon_train(self, d_optimizer, g_optimizer, epochs, dsteps_per_gstep, batch_size, printing = False):

        def _discriminator_loss(doutputs_true, doutputs_fake):
            losses_true = torch.log(doutputs_true)
            losses_fake = torch.log(1.0 - doutputs_fake) # 1.0 minus each element
            return -torch.mean(losses_true + losses_fake)

        # NOTE: this isn't combined with discriminator_loss to produce both losses simultaneously
        # is because for training the generator, we will resample the noise
        def _generator_loss(doutputs_fake):
            losses_fake = torch.log(1.0 - doutputs_fake) # 1.0 minus each element
            return torch.mean(losses_fake)

        print("Training GAN: ", end="")
        progress_bar_width = 50
        # setup toolbar
        sys.stdout.write("[%s]" % (" " * progress_bar_width))
        sys.stdout.flush()
        sys.stdout.write("\b" * (progress_bar_width+1)) # return to start of line, after '['

        for epoch in range(epochs):
            if epoch%(int(epochs/progress_bar_width)) == 0:
                sys.stdout.write("#")
                sys.stdout.flush()

            if printing: print("epoch {}...".format(epoch))

            # ------------------------
            # Train the discriminator
            # ------------------------
            for _ in range(dsteps_per_gstep):
                d_optimizer.zero_grad()

                # train on real data
                true_batch = self.discriminator.generate_true_samples(batch_size)
                true_outputs = self.discriminator.forward(true_batch)

                # train on fake data
                fake_batch = self.generator.generate_samples(batch_size)
                fake_outputs = self.discriminator.forward(fake_batch)
                
                # compute the loss and take optimizer step
                d_loss = _discriminator_loss(true_outputs, fake_outputs)
                d_loss.backward()
                d_optimizer.step()

            # ------------------------
            # Train the generator
            # ------------------------
            g_optimizer.zero_grad()

            # generate samples
            fake_batch = self.generator.generate_samples(batch_size)
            fake_outputs = self.discriminator.forward(fake_batch)

            # compute the loss and take optimizer step
            g_loss = _generator_loss(fake_outputs)
            g_loss.backward()
            g_optimizer.step()

        sys.stdout.write("\n")

    # train using wasserstein L1 distance (see vanilla WGAN) and naive clipping procedure
    # to maintain Lipschitz condition
    def wasserstein_train_basic(self, d_optimizer, g_optimizer, epochs, dsteps_per_gstep, batch_size, clipping):
        pass # TODO

    # also train using wasserstein L1 distance but with https://arxiv.org/abs/1704.00028
    def wasserstein_train_adv(self, d_optimizer, g_optimizer, epochs, dsteps_per_gstep, batch_size):
        pass # TODO

    # train using least squares divergence (see LSGAN)
    def least_squares_train(self, d_optimizer, g_optimizer, epochs, dsteps_per_gstep, batch_size):
        pass # TODO

### Generator for simple distributions ###
# NOTE: architecture optimize for 1D sampling (smaller network for simple 1D distributions)
class DistributionGenerator(nn.Module):
    # currently just a basic neural network with one hidden layer
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
        x = self.fc1_nonlin(self.fc1_lin(x))
        x = self.fc2_nonlin(self.fc2_lin(x))
        x = self.fc3_lin(x)
        return x

    def generate_samples(self, batch_size):
        noise_samples = Variable(torch.FloatTensor([self.input_noise_distribution() for _ in range(batch_size)]))
        return self.forward(noise_samples)

### Discriminator for simple distributions ###
class DistributionDiscriminator(nn.Module):
    # currently just a basic neural network with one hidden layer
    def __init__(self, target_distribution, input_dim, hidden_dim):
        super(DistributionDiscriminator, self).__init__()
        self.target_distribution = target_distribution
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
        
        self.make_prob = F.softmax

    # NOTE: let's use the convention that index 0 means NOT from true distribution and
    # index 1 means from true distribution
    def forward(self, x):
        x = self.fc1_nonlin(self.fc1_lin(x))
        x = self.fc2_nonlin(self.fc2_lin(x))
        x = self.make_prob(self.fc3_lin(x), dim=1)
        return x.select(1, 1) # probability of sample coming from truth

    def discriminate_sample(self, candidate_sample):
        probs = self.forward(candidate_sample)
        return probs.argmax()

    def generate_true_samples(self, batch_size):
        return Variable(torch.FloatTensor([self.target_distribution() for _ in range(batch_size)]))


### Generator for images ###
class ImageGenerator(nn.Module):

  # convolutional neural network
  def __init__(self, input_height, input_width, input_channels, output_dim, conv_dim):
    super(CNN, self).__init__()
    # pass these in to make it more "customizable"
    self.input_height = input_height
    self.input_width = input_width
    self.output_dim = output_dim
    self.conv_dim = conv_dim
    self.input_channels = input_channels
    
    # some things to play with
    self.fc1_out_dim = 30
    self.fc2_out_dim = 30
    self.conv1_output_channels = 3
    self.conv2_output_channels = 5
    
    self.dim_after_convs = (self.conv2_output_channels * (input_height - 2 * (conv_dim - 1)) ** 2)

    self.conv1 = nn.Conv2d(input_channels, self.conv1_output_channels, conv_dim)
    #self.pool = nn.MaxPool2d(2, 2) # ignore for now
    self.conv2 = nn.Conv2d(self.conv1_output_channels, self.conv2_output_channels, conv_dim)
    # NOTE: fc stands for "fully-connected
    self.conv_nonlin = F.relu # cuz why not
    self.fc_nonlin = F.elu # cuz why not
    self.fc1 = nn.Linear(self.dim_after_convs, self.fc1_out_dim)
    self.fc2 = nn.Linear(self.fc1_out_dim, self.fc2_out_dim)
    self.fc3 = nn.Linear(self.fc2_out_dim, output_dim)

  def forward(self, x):
    # x = self.pool(self.conv_nonlin(self.conv1(x)))
    # x = self.pool(self.conv_nonlin(self.conv2(x)))
    x = self.conv_nonlin(self.conv1(x))
    x = self.conv_nonlin(self.conv2(x))
    x = x.view(-1, self.dim_after_convs) # compress into a vector
    x = self.fc_nonlin(self.fc1(x))
    x = self.fc_nonlin(self.fc2(x))
    x = self.fc3(x)
    return x

### Discriminator for images ###
class ImageDiscriminator(nn.Module):

  # convolutional neural network
  def __init__(self, input_height, input_width, input_channels, conv_dim):
    super(CNN, self).__init__()
    # pass these in to make it more "customizable"
    self.input_height = input_height
    self.input_width = input_width
    self.output_dim = 2
    self.conv_dim = conv_dim
    self.input_channels = input_channels
    
    # some things to play with
    self.fc1_out_dim = 30
    self.fc2_out_dim = 30
    self.conv1_output_channels = 3
    self.conv2_output_channels = 5
    
    self.dim_after_convs = (self.conv2_output_channels * (input_height - 2 * (conv_dim - 1)) ** 2)

    self.conv1 = nn.Conv2d(input_channels, self.conv1_output_channels, conv_dim)
    self.conv2 = nn.Conv2d(self.conv1_output_channels, self.conv2_output_channels, conv_dim)
    # NOTE: fc stands for "fully-connected
    self.conv_nonlin = F.relu # cuz why not
    self.fc_nonlin = F.elu # cuz why not
    self.fc1 = nn.Linear(self.dim_after_convs, self.fc1_out_dim)
    self.fc2 = nn.Linear(self.fc1_out_dim, self.fc2_out_dim)
    self.fc3 = nn.Linear(self.fc2_out_dim, output_dim)

  def forward(self, x):
    x = self.conv_nonlin(self.conv1(x))
    x = self.conv_nonlin(self.conv2(x))
    x = x.view(-1, self.dim_after_convs) # compress into a vector
    x = self.fc_nonlin(self.fc1(x))
    x = self.fc_nonlin(self.fc2(x))
    x = nn.softmax(self.fc3(x))
    return x

def build_dist_gan(noise_dist,		# sampler for noise distribution
				   target_dist,		# sampler for target distribution
				   noise_dim,		# dim of the noise_dist
				   target_dim,		# dim of the target dist
				   gen_hidden_dim,	# hidden dimension size for the generator network
				   dis_hidden_dim):	# hidden dimension size for the discriminator network		 
	dist_generator = DistributionGenerator(noise_dist, noise_dim, target_dim, gen_hidden_dim)
	dist_discriminator = DistributionDiscriminator(target_dist, target_dim, dis_hidden_dim)
	return GAN(dist_generator, dist_discriminator)

def build_image_gan():
	pass