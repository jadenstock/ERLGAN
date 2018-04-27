# other utils
import numpy as np
#import multiprocessing as mp
import sys
#from scipy.stats import entropy
#from numpy.linalg import norm

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

    def generate_samples(self, n):
        noise_samples = Variable(torch.FloatTensor([self.input_noise_distribution() for _ in range(n)]))
        return self.forward(noise_samples)

class DistributionDiscriminator(nn.Module):
    # currently just a basic neural network with one hidden layer
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

class DistributionGAN:

    def __init__(self,
                 input_noise_distribution,  # a function of no arguments that produces a noise vector
                 target_distribution,       # a function of no arguments that samples from the target distribution
                 noise_dim,                 # the dimension of noise to generator
                 sample_dim,                # the dimension of generator output/discriminator input
                 gen_hidden_dim,            
                 dis_hidden_dim):
        self.input_noise_distribution = input_noise_distribution
        self.target_distribution = target_distribution
        self.generator = DistributionGenerator(input_noise_distribution, noise_dim, sample_dim, gen_hidden_dim)
        self.discriminator = DistributionDiscriminator(sample_dim, dis_hidden_dim)

    # train using vanilla Jensen-Shannon divergence/KL divergence (see vanilla GAN)
    # (see also https://github.com/devnag/pytorch-generative-adversarial-networks/blob/master/gan_pytorch.py)
    def jensen_shannon_train(self, d_optimizer, g_optimizer, epochs, dsteps_per_gstep, batch_size, printing = False):

        def discriminator_loss(doutputs_true, doutputs_fake):
            losses_true = torch.log(doutputs_true)
            losses_fake = torch.log(1.0 - doutputs_fake) # 1.0 minus each element
            return -torch.mean(losses_true + losses_fake)

        # NOTE: this isn't combined with discriminator_loss to produce both losses simultaneously
        # is because for training the generator, we will resample the noise
        def generator_loss(doutputs_fake):
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
                true_batch = Variable(torch.FloatTensor([self.target_distribution() for _ in range(batch_size)]))
                true_outputs = self.discriminator.forward(true_batch)

                # train on fake data
                noise_samples = Variable(torch.FloatTensor([self.input_noise_distribution() for _ in range(batch_size)]))
                fake_batch = self.generator(noise_samples)
                fake_outputs = self.discriminator.forward(fake_batch)
                
                # compute the loss and take optimizer step
                d_loss = discriminator_loss(true_outputs, fake_outputs)
                d_loss.backward()
                d_optimizer.step()

            # ------------------------
            # Train the generator
            # ------------------------
            g_optimizer.zero_grad()

            # generate samples and perform an update
            fake_batch = self.generator.generate_samples(batch_size)
            fake_outputs = self.discriminator.forward(fake_batch)
            g_loss = generator_loss(fake_outputs)
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

class ImageDiscriminator(nn.Module):

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

class ConvolutionGAN:

  def __init__(self,
                 input_noise_distribution,  # a function of size -> vector
                 target_distribution,       # a function of size -> vector
                 noise_dim,                 # the dimension of noise to generator
                 sample_dim,                # the dimension of generator output/discriminator input
                 gen_hidden_dim,            
                 dis_hidden_dim):
        self.input_noise_distribution = input_noise_distribution
        self.target_distribution = target_distribution
        self.generator = ImageGenerator()

        # input_height, input_width, input_channels, output_dim, conv_dim
        self.discriminator = ImageDiscriminator()

  def jensen_shannon_train(self, d_optimizer, g_optimizer, epochs, dsteps_per_gstep, batch_size, printing = False):
    pass

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
        noise_samples = Variable(torch.FloatTensor([input_noise_distribution() for _ in range(num_samples)]))
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

if __name__ == "__main__":
  dist = "dist" in sys.argv    # train distribution gan
  image = "image" in sys.argv  # train image gan

  if dist:
    print("Setting up Distribution GAN...")
    noise_dim = 1 # dimensionality of noise distribution
    sample_dim = 1 # dimensionality of generator output (and target distribution)
    gen_hidden_dim = 20 # hidden layer size for generator
    dis_hidden_dim = 20 # hidden layer size for discriminator
    df = 4 # for Chi dist.
    epochs = 1000 # number of training epochs
    dsteps_per_gstep = 5 # TODO: adaptive dsteps_per_gstep
    batch_size = 10 # bacth size per step

    input_noise_distribution = lambda : np.random.normal(loc=0.0, scale=1.0, size=noise_dim)
    target_distribution = lambda : np.random.chisquare(df=df, size=sample_dim)

    gan = DistributionGAN(input_noise_distribution, target_distribution,
                          noise_dim, sample_dim, gen_hidden_dim, dis_hidden_dim)

    # train
    d_optimizer = optim.SGD(gan.discriminator.parameters(), lr=0.001, momentum=0.9)
    g_optimizer = optim.SGD(gan.generator.parameters(), lr=0.001, momentum=0.9)
    gan.jensen_shannon_train(d_optimizer, g_optimizer, epochs, dsteps_per_gstep, batch_size, printing=False)

    # generate
    num_samples_list = [10000]
    tv_dists, p_apxs = test_gan_efficiency(input_noise_distribution, gan.generator, target_distribution, num_samples_list)
    _, tv_dist = next(iter(tv_dists.items()))
    _, (p_g_apx, p_t_apx) = next(iter(p_apxs.items()))
    p_g_xs, p_g_ys = zip(*p_g_apx.items())
    p_t_xs, p_t_ys = zip(*p_t_apx.items())

    # plot
    plt.scatter(p_g_xs, p_g_ys, color='b')
    plt.scatter(p_t_xs, p_t_ys, color='r')
    plt.savefig("chi_squared_gan_vis.png")

    # tv distance
    print(tv_dist)

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
    # TODO: train image gan

