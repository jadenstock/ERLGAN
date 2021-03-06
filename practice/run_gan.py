# other utils
import numpy as np
import sys
from collections import Counter
from scipy import stats
from time import time

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
  args = set(sys.argv)
  dist = "dist" in args    # train distribution gan
  image = "image" in args  # train image gan
  save = "save" in args    # whether or not to save produced models
  load = "load" in args    # whether or not to load pretrained models
  train = "train" in args  # whether or not to do training

  model_dir = "./models/"
  image_dir = "./images/"

  if dist:
    if load:
      dist_gan = load_gan("gaussian_chi_sq_dist_gan", model_dir)
    else:
      print("Setting up Distribution GAN for Training...")
      noise_dim = 1 # dimensionality of noise distribution
      sample_dim = 1 # dimensionality of generator output (and target distribution)
      gen_hidden_dim = 10 # hidden layer size for generator
      dis_hidden_dim = 10 # hidden layer size for discriminator
      df = 4 # for Chi dist.
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
    if train:
      epochs = 5 # number of training epochs
      dsteps_per_gstep = 5 # TODO: adaptive dsteps_per_gstep
      lr_sgd = 0.001
      lr_adam = 0.0001
      eval_sample_size = 250
#      g_optimizer = optim.SGD(dist_gan.generator.parameters(), lr=lr_sgd, momentum=0.9)
#      d_optimizer = optim.SGD(dist_gan.discriminator.parameters(), lr=lr_sgd, momentum=0.9)

      g_optimizer = optim.Adam(dist_gan.generator.parameters(), lr=lr_adam)
      d_optimizer = optim.Adam(dist_gan.discriminator.parameters(), lr=lr_adam)

      start_time = time()
      dist_gan.jensen_shannon_train(d_optimizer, g_optimizer, epochs, dsteps_per_gstep, eval_sample_size, printing=False)
      end_time = time()
      print("Completed Training in {} Seconds...".format(end_time - start_time))

    # generate
    num_samples_list = [1000, 10000, 100000]
    tv_dists, p_apxs = test_gan_efficiency(dist_gan.generator.noise_dist_sampler,
                                           dist_gan.generator,
                                           dist_gan.discriminator.target_dist_sampler,
                                           num_samples_list)
#    _, tv_dist = next(iter(tv_dists.items()))
#    _, (p_g_apx, p_t_apx) = next(iter(p_apxs.items()))
    p_g_apx, p_t_apx = p_apxs[num_samples_list[2]]

    # plot
    plot_gan_vs_true_distribution_1D(p_g_apx, p_t_apx, image_dir, "chi_squared_gan_vis.png")

    # tv distance
    print(tv_dists)

    if save:
      save_gan(dist_gan, model_dir)

  if image:
    noise_dim = 100
    if load:
      image_gan = load_gan("gaussian_mnist_image_gan", model_dir)
    else:
      print("Setting up Image GAN for Training...")
      gen_hidden_dims = [20, 20] # hidden layer sizes for generator
      dis_hidden_dims = [20, 20] # hidden layer sizes for discriminator
      batch_size = 10 # bacth size per step
      kernel_dim = 4 # for discriminator convolutions and generator deconvolutions

      ### code special for mnist ###
      transform = transforms.Compose([transforms.ToTensor(), binarization_transform])
      mnist_train_set = torchvision.datasets.MNIST(root="./mnist",
                                                   train=True,
                                                   download=True,
                                                   transform=transform)
      # NOTE: useful for getting a subset just to test if the code works
      # Need to comment out to do proper training
#      mnist_train_set = DatasetIntervalSubset(mnist_train_set, 0, 10000)
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
    if train:
      epochs = 5 # number of training epochs
      dsteps_per_gstep = 5 # TODO: adaptive dsteps_per_gstep

      g_optimizer = optim.SGD(image_gan.generator.parameters(), lr=0.001, momentum=0.9)
      d_optimizer = optim.SGD(image_gan.discriminator.parameters(), lr=0.001, momentum=0.9)

      start_time = time()
      image_gan.jensen_shannon_train(d_optimizer, g_optimizer, epochs, dsteps_per_gstep, printing=False)
      end_time = time()
      print("Completed Training in {} Seconds...".format(end_time - start_time))

    n_samples = 10
    true_example_pixels = image_gan.discriminator.generate_true_samples(n_samples).data.numpy()
    gen_example_pixels = image_gan.generator.generate_samples(n_samples).data.numpy()
    for i in range(n_samples):
      generate_grayscale_image(np.squeeze(true_example_pixels[i]), image_dir, "true_example_{}.png".format(i))
      generate_grayscale_image(np.squeeze(gen_example_pixels[i]), image_dir, "generator_example_{}.png".format(i))

    if save:
      save_gan(image_gan, model_dir)

