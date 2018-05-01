# other utils
import numpy as np
import sys
import os
from collections import Counter
from scipy import stats

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

# all gans
from gan import *

# NOTE can be useful for pickling lambda functions
import dill

##### Model Evaluation Utilities #####

# NOTE: only works for finite probability distributions
# Takes as input two maps which map element to probability.
def total_variation_distance(p, q):
  Omega = set()
  Omega = Omega | p.keys() | q.keys()
  dist = 0.0
  for omega in Omega:
    if omega in p and omega in q:
      dist += abs(p[omega] - q[omega])
    elif omega in p:
      dist += p[omega]
    elif omega in q:
      dist += q[omega]
  return dist / 2.0

def generate_approximate_distribution_from_bins(binned_samples):
  if type(binned_samples) is not list:
    binned_samples = list(binned_samples)
  p_apx = dict(Counter(binned_samples))
  s = float(sum(p_apx.values()))
  for key in p_apx.keys():
    p_apx[key] = p_apx[key] / s
  return p_apx

# NOTE: currently only works for 1D samples
def test_gan_efficiency(noise_distribution, generator, target_distribution, num_samples_list, bin_decimal_places=1):
  tv_dists = {} # for measuring total variation distance
  p_apxs = {} # useful for plotting
  for num_samples in num_samples_list:
    noise_samples = Variable(torch.FloatTensor([noise_distribution() for _ in range(num_samples)]))
    generator_samples = generator(noise_samples).data.numpy()
    generator_samples = np.array([arr[0] for arr in generator_samples])
    binned_generator_samples = np.around(generator_samples, decimals=bin_decimal_places)
    p_g_apx = generate_approximate_distribution_from_bins(binned_generator_samples)

    target_samples = np.array([target_distribution()[0] for _ in range(num_samples)])
    binned_target_samples = np.around(target_samples, decimals=bin_decimal_places)
    p_t_apx = generate_approximate_distribution_from_bins(binned_target_samples)
    tv_dist = total_variation_distance(p_g_apx, p_t_apx)

    tv_dists[num_samples] = tv_dist
    p_apxs[num_samples] = (p_g_apx, p_t_apx)
  return tv_dists, p_apxs

# https://stats.stackexchange.com/questions/244012/can-you-explain-parzen-window-kernel-density-estimation-in-laymans-terms/244023
#
# Kernel estimation is an alternative to binning the samples into a histogram.
# This is the method that the original GAN paper uses and seems to be a
# standard technique. Unfortunately estimates can have high variance so we
# need to be careful.
def gaussian_parzen_NLL_estimate(fit_data, predict_data):
  # handles case when we're estimating a 1D distribution
  if fit_data.shape[1] == 1:
    fit_data = fit_data.squeeze(1)
  if predict_data.shape[1] == 1:
    predict_data = predict_data.squeeze(1)
  # TODO: implement cross validation to select bandwidth
  bandwidth = 'silverman'
  kernel_model = stats.gaussian_kde(fit_data, bandwidth)
  return -1.0 * np.sum(kernel_model.logpdf(predict_data))

######################################

##### Plotting Utilities #####

def plot_gan_vs_true_distribution_1D(p_g_apx, p_t_apx, dir_path, filename, title=None):
  p_g_xs, p_g_ys = zip(*p_g_apx.items())
  p_t_xs, p_t_ys = zip(*p_t_apx.items())
  plt.scatter(p_g_xs, p_g_ys, color='b', label='GAN')
  plt.scatter(p_t_xs, p_t_ys, color='r', label='Truth')
  if title is not None:
    plt.title(title)
  plt.xlabel("Binned Samples")
  plt.ylabel("Bin Probabilities")
  plt.legend()
  plt.savefig(os.path.join(dir_path, filename))

def generate_grayscale_image(pixels, dir_path, filename):
  plt.imshow(pixels, cmap="gray")
  plt.savefig(os.path.join(dir_path, filename))

##############################

##### Model Saving and Loading Utilities #####

# this is the "recommended" way for saving models
def save_gan_parameters(gan, dir_path):
  torch.save(gan.generator.state_dict(), os.path.join(dir_path, gan.name + ".generator_parameters.pth"))
  torch.save(gan.discriminator.state_dict(), os.path.join(dir_path, gan.name + ".discriminator_parameters.pth"))

# TODO: implement loading gan parameters
# main challenge is we don't know which type
# of generator or dscriminator we have so
# it's hard to initialize the model that will eventually call .load_state_dict

# for now, do quick and dirty. also see
# https://github.com/pytorch/text/issues/73
# for torch saving a model that has other fields
def save_gan(gan, dir_path):
  torch.save(gan.generator, os.path.join(dir_path, gan.name + ".generator.pth"), pickle_module=dill)
  torch.save(gan.discriminator, os.path.join(dir_path, gan.name + ".discriminator.pth"), pickle_module=dill)

def load_gan(name, dir_path):
  g = None
  d = None
  try:
    g = torch.load(os.path.join(dir_path, name + ".generator.pth"), pickle_module=dill)
  except:
    print("Could not load generator \"{}\" from {}".format(name, dir_path))
  try:
    d = torch.load(os.path.join(dir_path, name + ".discriminator.pth"), pickle_module=dill)
  except:
    print("Could not load discriminator \"{}\" from {}".format(name + ".discriminator.pth", dir_path))
  if g is not None and d is not None:
    return GAN(g, d, name)
  else:
    return None

##############################################

