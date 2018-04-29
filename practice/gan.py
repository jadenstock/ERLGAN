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
  def jensen_shannon_train(self, d_optimizer, g_optimizer, epochs, dsteps_per_gstep, printing = False):

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
    # setup progress bar
    progress_bar_width = 50
    sys.stdout.write("[%s]" % (" " * progress_bar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (progress_bar_width+1)) # return to start of line, after '['

    # set up local counters/variables for updating progress bar
    total_discrim_samples = epochs * len(self.discriminator.target_dist_loader.dataset)
    discrim_samples_per_bar = int(total_discrim_samples / progress_bar_width)
    discrim_samples_used = 0
    batch_size = self.discriminator.target_dist_loader.batch_size

    for epoch in range(epochs):

      if printing: print("epoch {}...".format(epoch))

      # -----------------------
      # Train the discriminator
      # -----------------------
      for i, batch in enumerate(self.discriminator.target_dist_loader, 1):
        true_batch, _ = batch
        true_batch = Variable(true_batch)
        true_outputs = self.discriminator.forward(true_batch)

        fake_batch = self.generator.generate_samples(batch_size)
        fake_outputs = self.discriminator.forward(fake_batch)
        
        d_optimizer.zero_grad()

        # compute the loss and take optimizer step
        d_loss = _discriminator_loss(true_outputs, fake_outputs)
        d_loss.backward()
        d_optimizer.step()

        # -------------------
        # Train the generator
        # -------------------
        if i % dsteps_per_gstep == 0:  
          g_optimizer.zero_grad()

          # generate samples
          fake_batch = self.generator.generate_samples(batch_size)
          fake_outputs = self.discriminator.forward(fake_batch)

          # compute the loss and take optimizer step
          g_loss = _generator_loss(fake_outputs)
          g_loss.backward()
          g_optimizer.step()

        # means at some point in between, we reached a nujmber of
        # discriminator samples used that is evenly divisible by
        # discrim_samples_per_bar. Thus, we can add a new bar.
        # This handles the case when the batch_size doesn't nicely
        # match up with the discim_samples_per_bar.
        sys.stdout.write("#" * int((batch_size / discrim_samples_per_bar) - 1))
        if (discrim_samples_used + batch_size) % discrim_samples_per_bar <= \
           discrim_samples_used % discrim_samples_per_bar:
          sys.stdout.write("#")
          sys.stdout.flush()
        
        discrim_samples_used += batch_size
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
  def __init__(self, noise_dist_sampler, input_dim, output_dim, hidden_dim):
    super(DistributionGenerator, self).__init__()
    self.noise_dist_sampler = noise_dist_sampler
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
    noise_samples = Variable(torch.FloatTensor([self.noise_dist_sampler() for _ in range(batch_size)]))
    return self.forward(noise_samples)

### Discriminator for simple distributions ###
class DistributionDiscriminator(nn.Module):
  # currently just a basic neural network with one hidden layer
  def __init__(self, target_dist_loader, input_dim, hidden_dim):
    super(DistributionDiscriminator, self).__init__()
    self.target_dist_loader = target_dist_loader
    self.target_dist_sampler = target_dist_loader.dataset.dist_sampler
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
    return Variable(torch.FloatTensor([self.target_dist_sampler() for _ in range(batch_size)]))

### Generator for images ###
class ImageGenerator(nn.Module):

  # "deconvolutional" network for generating images given noise
  # NOTE: hidden_dims must have length equal to number of hidden layers minus one
  def __init__(self, noise_dist_sampler, input_dim, output_width, output_height, output_channels, hidden_dims, kernel_dim):
    super(ImageGenerator, self).__init__()
    # pass these in to make it more "customizable"
    self.noise_dist_sampler = noise_dist_sampler
    self.input_dim = input_dim
    self.output_channels = output_channels
    self.output_width = output_width
    self.output_height = output_height
    self.hidden_dims = hidden_dims
    self.kernel_dim = kernel_dim

    # compute dimension of last hidden layer before deconvolution
    self.pre_deconv_width = output_width - 2 * (kernel_dim - 1)
    self.pre_deconv_height = output_height - 2 * (kernel_dim - 1)
    
    # NOTE: fc stands for "fully-connected
    self.fc1_nonlin = F.elu # cuz why not
    self.fc2_nonlin = F.relu # cuz why not
    self.fc1 = nn.Linear(input_dim, hidden_dims[0])
    self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
    self.fc3 = nn.Linear(hidden_dims[1], self.pre_deconv_width * self.pre_deconv_height)

    # "deconvolution" for producing the images
    self.deconv1 = nn.ConvTranspose2d(1, output_channels, kernel_dim)
    self.deconv2 = nn.ConvTranspose2d(output_channels, output_channels, kernel_dim)

  def forward(self, x):
    x = self.fc1_nonlin(self.fc1(x))
    x = self.fc2_nonlin(self.fc2(x))
    x = self.fc3(x)
    x = x.view(-1, 1, self.pre_deconv_width, self.pre_deconv_height)
    x = self.deconv1(x)
    x = self.deconv2(x)
    return x

  def generate_samples(self, batch_size):
    noise_samples = Variable(torch.FloatTensor([self.noise_dist_sampler() for _ in range(batch_size)]))
    return self.forward(noise_samples)

### Discriminator for images ###
class ImageDiscriminator(nn.Module):

  # convolutional neural network
  def __init__(self, target_dist_loader, input_height, input_width, input_channels, hidden_dims, kernel_dim):
    super(ImageDiscriminator, self).__init__()

    self.target_dist_loader = target_dist_loader

    # pass these in to make it more "customizable"
    self.input_height = input_height
    self.input_width = input_width
    self.output_dim = 2
    self.kernel_dim = kernel_dim
    self.input_channels = input_channels
    
    # some things to play with
    self.fc1_out_dim = hidden_dims[0]
    self.fc2_out_dim = hidden_dims[1]
    self.conv1_output_channels = 3
    self.conv2_output_channels = 5
    
    self.dim_after_convs = (self.conv2_output_channels * (input_height - 2 * (kernel_dim - 1)) ** 2)

    self.conv1 = nn.Conv2d(input_channels, self.conv1_output_channels, kernel_dim)
    self.conv2 = nn.Conv2d(self.conv1_output_channels, self.conv2_output_channels, kernel_dim)
    # NOTE: fc stands for "fully-connected
    self.conv_nonlin = F.relu # cuz why not
    self.fc_nonlin = F.elu # cuz why not
    self.fc1 = nn.Linear(self.dim_after_convs, self.fc1_out_dim)
    self.fc2 = nn.Linear(self.fc1_out_dim, self.fc2_out_dim)
    self.fc3 = nn.Linear(self.fc2_out_dim, self.output_dim)

    self.make_prob = F.softmax

  def forward(self, x):
    x = self.conv_nonlin(self.conv1(x))
    x = self.conv_nonlin(self.conv2(x))
    x = x.view(-1, self.dim_after_convs) # compress into a vector
    x = self.fc_nonlin(self.fc1(x))
    x = self.fc_nonlin(self.fc2(x))
    x = self.make_prob(self.fc3(x), dim=1)
    return x.select(1, 1)

  def discriminate_sample(self, candidate_sample):
    probs = self.forward(candidate_sample)
    return probs.argmax()

  def generate_true_samples(self, batch_size):
    return Variable(torch.FloatTensor([self.target_dist_loader.next() for _ in range(batch_size)]))

def build_dist_gan(noise_dist_sampler,    # sampler for noise distribution
                   target_dist_loader,    # target distribution dataset loader
                   noise_dim,    # dim of the noise distribution
                   target_dim,    # dim of the target distribution
                   gen_hidden_dim,  # hidden dimension size for the generator network
                   dis_hidden_dim):  # hidden dimension size for the discriminator network
  dist_generator = DistributionGenerator(noise_dist_sampler, noise_dim, target_dim, gen_hidden_dim)
  dist_discriminator = DistributionDiscriminator(target_dist_loader, target_dim, dis_hidden_dim)
  return GAN(dist_generator, dist_discriminator)

def build_image_gan(noise_dist_sampler, # sample for noise distribution
                    target_dist_loader, # target distribution
                    noise_dim, # dim of noise distribution
                    image_height, # image height of generator output/discriminator input
                    image_width, # image width of generator output/discriminator input
                    channels, # number of channels of generator output/discriminator input
                    generator_hidden_dims, # hidden layers dimensions for generator
                    discriminator_hidden_dims, # hidden layers dimensions for discriminator
                    kernel_dim): # kernel dimensions for conv/deconv

  #input height and width for discriminator must match the output height and width for the generator
  discriminator_input_height = generator_output_height = image_height
  discriminator_input_width = generator_output_width = image_width
  discriminator_input_channels = generator_output_channels = channels

  image_discriminator = ImageDiscriminator(target_dist_loader,
                                           discriminator_input_height,
                                           discriminator_input_width,
                                           discriminator_input_channels,
                                           discriminator_hidden_dims,
                                           kernel_dim)

  image_generator = ImageGenerator(noise_dist_sampler,
                                   noise_dim,
                                   generator_output_width,
                                   generator_output_height,
                                   generator_output_channels,
                                   generator_hidden_dims,
                                   kernel_dim)

  return GAN(image_generator, image_discriminator)

