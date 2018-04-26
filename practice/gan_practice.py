# other utils
import numpy as np
import multiprocessing as mp
import sys
from scipy.stats import entropy
from numpy.linalg import norm

# torch autograd, nn, etc.
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DistributionGenerator(nn.Module):

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

    def generate_sample(self):
        x = self.input_noise_distribution(input_size)
        return self.forward(x)

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
        
        self.make_prob = nn.Softmax()

    # NOTE: let's use the convention that index 0 means NOT from true distribution and
    # index 1 means from true distribution
    def forward(self, x):
        x = self.fc1_nonlin(self.fc1_lin(x))
        x = self.fc2_nonlin(self.fc2_lin(x))
        x = self.make_prob(self.fc3_lin(x))
        return x.select(1, 1) # probability of sample coming from truth

    def discriminate_sample(self, candidate_sample):
        probs = self.forward(candidate_sample)
        return probs.argmax()

class DistributionGAN:

    def __init__(self,
                 input_noise_distribution,  # a function of size -> vector
                 target_distribution,       # a function of size -> vector
                 noise_dim,                 # the dimension of noise to generator
                 sample_dim,                # the dimension of generator output/discriminator input
                 gen_hidden_dim,            
                 dis_hidden_dim):
        self.input_noise_distribution = input_noise_distribution
        self.target_distribution = target_distribution
        self.noise_dim = noise_dim
        self.sample_dim = sample_dim
        self.generator = DistributionGenerator(input_noise_distribution, noise_dim, sample_dim, gen_hidden_dim)
        self.discriminator = DistributionDiscriminator(sample_dim, dis_hidden_dim)

    # train using vanilla Jensen-Shannon divergence/KL divergence (see vanilla GAN)
    # (see also https://github.com/devnag/pytorch-generative-adversarial-networks/blob/master/gan_pytorch.py)
    def jensen_shannon_train(self, d_optimizer, g_optimizer, epochs, dsteps_per_gstep, batch_size, printing = True):

        def discriminator_loss(doutputs_true, doutputs_fake):
            #print("Truth:{}".format(doutputs_true))
            #print("Fake:{}".format(doutputs_fake))
            losses_true = torch.log(doutputs_true)
            losses_fake = torch.log(1.0 - doutputs_fake) # 1.0 minus each element
            return -torch.mean(losses_true + losses_fake)

        # NOTE: this isn't combined with discriminator_loss to produce both losses simultaneously
        # is because for training the generator, we will resample the noise
        def generator_loss(doutputs_fake):
            losses_fake = torch.log(1.0 - doutputs_fake) # 1.0 minus each element
            return torch.mean(losses_fake)

        for epoch in range(epochs):
            if printing: print("epoch {}...".format(epoch), end="")
            # ------------------------
            # Train the discriminator
            # ------------------------
            for _ in range(dsteps_per_gstep):
                d_optimizer.zero_grad()

                # train on real data
                true_batch = Variable(torch.FloatTensor([self.target_distribution(self.sample_dim) for _ in range(batch_size)]))
                true_outputs = self.discriminator.forward(true_batch)

                # train on fake data
                noise_samples = Variable(torch.FloatTensor([self.input_noise_distribution(self.noise_dim) for _ in range(batch_size)]))
                fake_batch = self.generator(noise_samples)
                fake_outputs = self.discriminator.forward(fake_batch)
                
                # compute the loss and take optimizer step
                d_loss = discriminator_loss(true_outputs, fake_outputs)
                print("d_loss: {}".format(d_loss))
                d_loss.backward()
                d_optimizer.step()
            if printing: print("discriminator trained...", end="")

            # ------------------------
            # Train the generator
            # ------------------------
            g_optimizer.zero_grad()

            # generate samples
            noise_samples = Variable(torch.FloatTensor([self.input_noise_distribution(self.noise_dim) for _ in range(batch_size)]))
            fake_batch = self.generator(noise_samples)

            # generator update
            fake_outputs = self.discriminator.forward(fake_batch)
            g_loss = generator_loss(fake_outputs)
            print("g_loss: {}".format(g_loss))

            g_loss.backward()
            g_optimizer.step()
            
            if printing: print("generator trained...", end="")

            if printing: print("\nJensen-Shannon loss is {}".format(0))

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

if __name__ == "__main__":
    input_noise_distribution = lambda size : np.random.normal(loc=0.0, scale=1.0, size=size)
    target_distribution = lambda size : np.random.chisquare(df=10, size=size)

    noise_dim = 10 # dimensionality of noise distribution
    sample_dim = 10 # dimensionality of generator output (and target distribution)
    gen_hidden_dim = 20 # hidden layer size for generator
    dis_hidden_dim = 20 # hidden layer size for discriminator

    gan = DistributionGAN(input_noise_distribution, target_distribution,
                          noise_dim, sample_dim, gen_hidden_dim, dis_hidden_dim)

    d_optimizer = optim.SGD(gan.discriminator.parameters(), lr=0.001, momentum=0.9)
    g_optimizer = optim.SGD(gan.generator.parameters(), lr=0.001, momentum=0.9)
    # d_optimizer, g_optimizer, epochs, dsteps_per_gstep, batch_size
    gan.jensen_shannon_train(d_optimizer, g_optimizer, 1000, 1, 10)

