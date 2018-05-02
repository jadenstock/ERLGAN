import numpy as np

import torch
import torchvision
import torch.utils.data as data

def binarize_image_tensor(image_tensor, dim=0, threshold=0.5):
  thresh = lambda x : np.float32(x <= threshold)
  return torch.FloatTensor(thresh(image_tensor.squeeze().numpy())).unsqueeze(dim)

binarization_transform = torchvision.transforms.Lambda(lambda x : binarize_image_tensor(x))
squeeze_transform = torchvision.transforms.Lambda(lambda x : x.view(x.size()[0] * x.size()[1]))

# These dataset subset classes are useful for downsizing an entire dataset
# to speed up training. This is useful if you want to train a bad model
# quickly just to verify that the code itself all works.
class DatasetIntervalSubset(data.Dataset):

  # 1. Inclusive min_idx and exclusive of max_idx
  # 2. Requires 0 <= min_idx <= max_idx <= len(dataset)
  def __init__(self, dataset, min_idx, max_idx):
    self.dataset = dataset
    self.min_idx = min_idx
    self.max_idx = max_idx

  def __len__(self):
    return self.max_idx - self.min_idx

  def __getitem__(self, idx):
    return self.dataset[self.min_idx + idx]

# NOTE: this is space inefficient as for most purposes, especially
# if the ordering of the examples in the dataset are sufficiently
# well randomized, then an interval subset is sufficient and one
# does not need to store a big list of indices
#
# Use interval subset when possible
class DatasetSubset(data.Dataset):

  # indices must be a subset of range(len(dataset))
  def __init__(self, dataset, indices):
    self.dataset = dataset
    self.indices = indices

  def __len__(self):
    return len(self.indices)

  def __getitem__(self, idx):
    return self.dataset[self.indices[idx]]

# These "Dataset" classes are essentially wrappers for a regular
# distribution sampler. These are created for convenience so that
# we can wrap around them a torch DataLoader, which will be a more
# generic way to handle batch training.

# Static distribution dataset, which samples a fixed number of given
# samples initially and lets the user just cycle through them. The
# distribution is sampled the required number of times during construction
# and from then on, the samples are fixed for the DataLoader to cycle
# through them. This is basically inferior in all ways to the "dynamic"
# dataset below except running time of __getitem__ and the "sample
# complexity"
#
# Not sure when this would ever be useful apart from testing for our
# purposes (?)
class DistributionStaticSamplesDataset(data.Dataset):

  def __init__(self, dist_sampler, num_samples):
    self.dist_sampler = dist_sampler
    self.num_samples = num_samples
    self.samples = [self.dist_sampler() for _ in range(self.num_samples)]

  def __len__(self):
    return self.num_samples

  # samples will always be positive so always return 1 for label
  # note it can be treated as a dummy as discriminator has a loss
  def __getitem__(self, idx):
    return torch.FloatTensor(self.samples[idx]), 1

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

