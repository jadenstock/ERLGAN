# other utils
import numpy as np
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