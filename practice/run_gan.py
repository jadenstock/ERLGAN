# other utils
import numpy as np
import sys

# plotting
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from scipy import stats

from gan import *
from utils import *

if __name__ == "__main__":
  dist = "dist" in sys.argv    # train distribution gan
  image = "image" in sys.argv  # train image gan

  if dist:
    print("Setting up Distribution GAN...")
    noise_dim = 1 # dimensionality of noise distribution
    sample_dim = 1 # dimensionality of generator output (and target distribution)
    gen_hidden_dim = 10 # hidden layer size for generator
    dis_hidden_dim = 10 # hidden layer size for discriminator
    df = 4 # for Chi dist.
    epochs = 5000 # number of training epochs
    dsteps_per_gstep = 5 # TODO: adaptive dsteps_per_gstep
    batch_size = 10 # bacth size per step

    input_noise_distribution = lambda : np.random.normal(loc=0.0, scale=1.0, size=noise_dim)
    target_distribution = lambda : np.random.chisquare(df=df, size=sample_dim)

    dist_gan = build_dist_gan(input_noise_distribution, target_distribution, 
    					 noise_dim, sample_dim, gen_hidden_dim, dis_hidden_dim)

    # train
    d_optimizer = optim.SGD(dist_gan.discriminator.parameters(), lr=0.001, momentum=0.9)
    g_optimizer = optim.SGD(dist_gan.generator.parameters(), lr=0.001, momentum=0.9)
    dist_gan.jensen_shannon_train(d_optimizer, g_optimizer, epochs, dsteps_per_gstep, batch_size, printing=False)

    # generate
    num_samples_list = [1000, 10000, 100000]
    tv_dists, p_apxs = test_gan_efficiency(input_noise_distribution, dist_gan.generator, target_distribution, num_samples_list)
#    _, tv_dist = next(iter(tv_dists.items()))
#    _, (p_g_apx, p_t_apx) = next(iter(p_apxs.items()))
    p_g_apx, p_t_apx = p_apxs[num_samples_list[2]]
    p_g_xs, p_g_ys = zip(*p_g_apx.items())
    p_t_xs, p_t_ys = zip(*p_t_apx.items())

    # plot
    plt.scatter(p_g_xs, p_g_ys, color='b')
    plt.scatter(p_t_xs, p_t_ys, color='r')
    plt.savefig("chi_squared_gan_vis.png")

    # tv distance
    print(tv_dists)

  if image:
    print("Setting up Image GAN...")
    trainset = torchvision.datasets.MNIST(root="./mnist",
              train=True,
              download=True,
              transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset,
              batch_size=train_batch,
              shuffle=True, #shuffles the data? good
              num_workers=2)

    input_noise_distribution = lambda : np.random.normal(loc=0.0, scale=1.0, size=noise_dim)
    target_distribution = trainloader()

    image_gan = build_image_gan()
    # TODO: train image gan