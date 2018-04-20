import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data

class Generator(nn.Module):

	def __init__(self):
		super(Generator, self).__init__()
		# TODO

	def forward(self, x):
		# TODO

class Discriminator(nn.Module):

	def __init__(self):
		super(Discriminator, self).__init__()
		# TODO

	def forward(self, x):
		# TODO

class GAN(nn.Module):

	def __init__(self):
		super(GAN, self).__init__()
		self.generator = Generator() # TODO
		self.discriminator = Discriminator() # TODO

	def forward_generator(self, x):
		return self.generator.forward(x)

	def forward_discriminator(self, x):
		return self.discriminator.forward(x)

	def train_backprop(self):
		# TODO

	def train_ES(self):
		# TODO
