# other utils
import numpy as np
import sys
from collections import Counter
from scipy import stats

# torch data
import torchvision
import torchvision.transforms as transforms

# our code
from gan import *
from utils import *
from datasets import *

# fix to a number if needed to aid debug
seed = None
np.random.seed(seed)

if __name__ == "__main__":
  dist = "dist" in sys.argv    # train distribution gan
  image = "image" in sys.argv  # train image gan
  save = "save" in sys.argv    # whether or not to save produced models

  if dist:
    print("Setting up Distribution GAN...")
    noise_dim = 1 # dimensionality of noise distribution
    sample_dim = 1 # dimensionality of generator output (and target distribution)
    gen_hidden_dim = 10 # hidden layer size for generator
    dis_hidden_dim = 10 # hidden layer size for discriminator
    df = 4 # for Chi dist.
    epochs = 5 # number of training epochs
    dsteps_per_gstep = 5 # TODO: adaptive dsteps_per_gstep
    batch_size = 10 # bacth size per step

    num_samples = 50000
    noise_dist_sampler = lambda : np.random.normal(loc=0.0, scale=1.0, size=noise_dim)
    target_dist_sampler = lambda : np.random.chisquare(df=df, size=sample_dim)
    
    target_distribution = DistributionDynamicSamplesDataset(target_dist_sampler, num_samples)
    target_dist_loader = torch.utils.data.DataLoader(target_distribution,
                                                     batch_size=batch_size,
                                                     shuffle=True, #shuffles the data? good
                                                     num_workers=2)
    dist_gan = build_dist_gan(noise_dist_sampler, target_dist_loader, 
    			      noise_dim, sample_dim, gen_hidden_dim,
                              dis_hidden_dim,
                              "gaussian_chi_sq_dist_gan")

    # train
    g_optimizer = optim.SGD(dist_gan.generator.parameters(), lr=0.001, momentum=0.9)
    d_optimizer = optim.SGD(dist_gan.discriminator.parameters(), lr=0.001, momentum=0.9)
    dist_gan.jensen_shannon_train(d_optimizer, g_optimizer, epochs, dsteps_per_gstep, printing=False)

    # generate
    num_samples_list = [1000, 10000, 100000]
    tv_dists, p_apxs = test_gan_efficiency(noise_dist_sampler, dist_gan.generator, target_dist_sampler, num_samples_list)
#    _, tv_dist = next(iter(tv_dists.items()))
#    _, (p_g_apx, p_t_apx) = next(iter(p_apxs.items()))
    p_g_apx, p_t_apx = p_apxs[num_samples_list[2]]

    # plot
    plot_gan_vs_true_distribution_1D(p_g_apx, p_t_apx, "chi_squared_gan_vis.png")

    # tv distance
    print(tv_dists)

    if save:
      save_gan(dist_gan, "./models/")

  if image:
    print("Setting up Image GAN...")
    noise_dim = 100
    gen_hidden_dims = [20, 20] # hidden layer sizes for generator
    dis_hidden_dims = [20, 20] # hidden layer sizes for discriminator
    epochs = 5 # number of training epochs
    dsteps_per_gstep = 5 # TODO: adaptive dsteps_per_gstep
    batch_size = 10 # bacth size per step
    kernel_dim = 4 # for discriminator convolutions and generator deconvolutions

    ### code special for mnist ###
    mnist_train_set = torchvision.datasets.MNIST(root="./mnist",
                                                 train=True,
                                                 download=True,
                                                 transform=transforms.ToTensor())
    mnist_dist_loader = torch.utils.data.DataLoader(mnist_train_set,
                                                    batch_size=batch_size,
                                                    shuffle=True, #shuffles the data? good
                                                    num_workers=2)
    image_height = 28
    image_width = 28
    channels = 1
    ##############################

    noise_dist_sampler = lambda : np.random.normal(loc=0.0, scale=1.0, size=noise_dim)

    image_gan = build_image_gan(noise_dist_sampler,
                                mnist_dist_loader,
                                noise_dim,
                                image_height,
                                image_width,
                                channels,
                                gen_hidden_dims,
                                dis_hidden_dims,
                                kernel_dim,
                                "gaussian_mnist_image_gan")
    g_optimizer = optim.SGD(image_gan.generator.parameters(), lr=0.001, momentum=0.9)
    d_optimizer = optim.SGD(image_gan.discriminator.parameters(), lr=0.001, momentum=0.9)
    image_gan.jensen_shannon_train(d_optimizer, g_optimizer, epochs, dsteps_per_gstep, printing=False)

    if save:
      save_gan(image_gan, "./models/")

