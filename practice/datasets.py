import numpy as np

import torch
import torchvision
import torch.utils.data as data

# These "Dataset" classes are essentially wrappers for a regular
# distribution sampler. These are created for convenience so that
# we can wrap around them a torch DataLoader, which will be a more
# generic way to handle batch training.

# Static distribution dataset, which samples a fixed number of given
# samples initially and lets the user just cycle through them. The
# distribution is sampled the required number of times during construction
# and from then on, the samples are fixed for the DataLoader to cycle
# through them. This is basically inferior in all ways to the "dynamic"
# dataset below except running time of __getitem__.
#
# Not sure when this would ever be useful apart from testing (?)
class DistributionStaticSamplesDataset(data.Dataset):

  def __init__(self, dist_sampler, num_samples):
    self.dist_sampler = dist_sampler
    self.num_samples = num_samples
    self.samples = [self.dist_sampler() for _ in range(self.num_samples)]

  def __len__(self):
    return self.num_samples

  # samples will always be positive so always return 1 for label
  # note it can be treated as a dummy as discriminator has a loss
  def __getitem__(self, index):
    return torch.FloatTensor(self.samples[index]), 1

# Dynamic distribution dataset, which at each call of __getitem__
# returns a fresh sample probably never seen before. The parameter
# "index" doesn't actually do anything and is only there to ensure
# it interfaces well. "num_samples" is there for the same reason.
class DistributionDynamicSamplesDataset(data.Dataset):

  def __init__(self, dist_sampler, num_samples):
    self.dist_sampler = dist_sampler
    self.num_samples = num_samples

  def __len__(self):
    return self.num_samples

  # samples will always be positive so always return 1 for label
  # note it can be treated as a dummy as discriminator has a loss
  def __getitem__(self, _):
    return torch.FloatTensor(self.dist_sampler()), 1
