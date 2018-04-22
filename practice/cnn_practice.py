# other utils
import numpy as np
import multiprocessing as mp
import sys

# torch autograd, nn, etc.
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# torch data
import torchvision
import torchvision.transforms as transforms

##### DEFINING A CNN #####

class CNN(nn.Module):

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
    
    # NOTE: very confusing but basically this just makes sure that the output dimensions after convolutions
    # matches up with the linearities input dimensions applied after; this is needed for resizing the convolution
    # outputs into a batch of vectors. It depends on the convolution window size, the size of the
    # pooling windows, etc. and the number convolutions and pooling layers.
    #
    # TODO: Find a good way to handle this generically without hardcoding numbers. Ideally, write some function
    # that takes as input the parameters of the convolutions and pooling and spits out a number for the right
    # number of dimensions.
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
  #  x = self.pool(self.conv_nonlin(self.conv1(x)))
  #  x = self.pool(self.conv_nonlin(self.conv2(x)))
    x = self.conv_nonlin(self.conv1(x))
    x = self.conv_nonlin(self.conv2(x))
    x = x.view(-1, self.dim_after_convs) # compress into a vector
    x = self.fc_nonlin(self.fc1(x))
    x = self.fc_nonlin(self.fc2(x))
    x = self.fc3(x)
    return x

##### CODE FOR CNN TRAINING USING STANDARD BACKPROP #####

def train_net_via_backprop(net, trainloader, optimizer, criterion, epochs):
  for epoch in range(epochs):
    running_loss = 0.0
    for i, data_batch in enumerate(trainloader, 0):
      inputs, labels = data_batch # plural because of batch
      inputs, labels = Variable(inputs), Variable(labels)
      optimizer.zero_grad()
      outputs = net.forward(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      running_loss += loss.data[0]
      if i % 100 == (100 - 1):
        print("[Epoch {}, Iter {}] loss: {}".format(epoch + 1, i + 1, running_loss / 100))
        running_loss = 0.0

##### CODE FOR EVOLUTIONARY STRATEGIES #####

def score_perturbation(net, perturbation, inputs, labels, criterion, scores, index, std_dev):
  new_net = CNN(net.input_height, net.input_width, net.input_channels, net.output_dim, net.conv_dim)
  net_state_dict = net.state_dict()
  new_net.load_state_dict(make_perturbed_net_state_dict(net_state_dict, perturbation, std_dev))
  score = score_net_ES(new_net, inputs, labels, criterion)
  scores[index] = score

# always do N(0,1) independent entries
def get_net_perturbations(net_state_dict, seed): # may be useful to fix a seed to help debugging
  generator = np.random.RandomState(seed)
  perturbations = {}
  for name, params in net_state_dict.items():
    perturbations[name] = torch.from_numpy(generator.normal(loc=0.0, scale=1.0, size=params.size())).float()
  return perturbations

def make_perturbed_net_state_dict(net_state_dict, perturbation, std_dev):
  new_net = {}
  for name, params in net_state_dict.items():
    new_net[name] = params + std_dev * perturbation[name]
  return new_net

# TODO: this can actually be used for backprop above
def score_net_ES(net, inputs, labels, criterion):
  outputs = net.forward(inputs)
  loss = criterion(outputs, labels)
  return -loss.data[0] #/ inputs.size()[0] # inputs.size()[0] is the batch size

def average_nets_ES(net_state_dict, perturbations, scores, std_dev, lr):
  new_net = {}
  for name, params in net_state_dict.items():
    new_net[name] = (lr / (len(perturbations) * std_dev)) * sum([scores[i] * perturbations[i][name] for i in range(len(perturbations))])
    new_net[name].add_(params)
  return new_net

# TODO: NEED TO PARALLELIZE OR THIS WILL TAKE FOREVER
def train_net_via_ES(net, trainloader, std_dev, lr, pop_size, criterion, epochs, seed_generator_seed=42):
  seed_generator = np.random.RandomState(seed_generator_seed)
  max_seed = (2**31) - 1
  for epoch in range(epochs):
    for i, data_batch in enumerate(trainloader, 0):
      inputs, labels = data_batch
      inputs, labels = Variable(inputs), Variable(labels)
      net_state_dict = net.state_dict() 
      seeds = seed_generator.randint(low=0, high=max_seed, size=pop_size)
      perturbations = [get_net_perturbations(net_state_dict, seeds[j]) for j in range(pop_size)]
      
      # code not using multiprocessing
      scores = [1.0] * pop_size
      for j in range(pop_size):
        perturbation = perturbations[j]
        score_perturbation(net, perturbation, inputs, labels, criterion, scores, j, std_dev)
      
      """
      # code using multiprocessing
      manager = mp.Manager()
      scores = manager.list([1.0] * pop_size)
      processes = []
      for j in range(pop_size):
        perturbation = perturbations[j]
        p = mp.Process(target=score_perturbation, args=(net,
                                                        perturbation,
                                                        inputs,
                                                        labels,
                                                        criterion,
                                                        scores,
                                                        j,
                                                        std_dev))
        p.start()
        processes.append(p)
      for p in processes:
        p.join()
      """
      scores = np.array(scores)
      scores = scores / sum(scores) # TODO: normalization that isn't explicitly represented in original ES
      net.load_state_dict(average_nets_ES(net_state_dict, perturbations, scores, std_dev, lr))
      current_loss = score_net_ES(net, inputs, labels, criterion)
      if i % 10 == (10 - 1):
        print("[Epoch {}, Iter {}] loss: {}".format(epoch + 1, i + 1, current_loss))

##### TESTING CODE #####
def evaluate_net(net, testloader):
  correct = 0
  total = 0
  for data_batch in testloader:
    image, label = data_batch
    image, label = Variable(image), Variable(label)
    outputs = net.forward(image)
    _, prediction = torch.max(outputs.data, 1) # returns max and argmax :D
    total += 1
    if prediction[0] == label.data[0]:
      correct += 1
  print("{} correct out of {}".format(correct, total))

if __name__ == "__main__":
  # load data and set some basic parameters
  train_batch = 10
  if len(sys.argv) > 1 and sys.argv[1] == "ES": # default training to backprop
    # try larger batch size just for ES training as otherwise, too much
    # multiprocessing overhead
    train_batch = 100
  test_batch = 1
  epochs = 5
  es_lr = 0.1
  sgd_lr = 0.001
  es_std_dev = 0.1 # TODO: not really sure how to set this
  es_pop_size = 100 # TODO: also not really sure how to set this

  trainset = torchvision.datasets.MNIST(root="./mnist",
              train=True,
              download=True,
              transform=transforms.ToTensor())
  trainloader = torch.utils.data.DataLoader(trainset,
              batch_size=train_batch,
              shuffle=True,
              num_workers=2)
  testset = torchvision.datasets.MNIST(root="./mnist",
             train=False,
             download=True,
             transform=transforms.ToTensor())
  testloader = torch.utils.data.DataLoader(testset,
             batch_size=test_batch,
             shuffle=False,
             num_workers=2)

  classes = tuple([i for i in range(10)])

  net = CNN(28, 28, 1, 10, 4) # dimensions hardcoded for MNIST
  #net_state_dict = net.state_dict()
  #for name, params in net_state_dict.items():
  #  print("Params for {}: {}".format(name, params))

  # Literally, all this is is adding a softmax on the outputs of the neural network (because the outputs
  # are "weights" corresponding to each class) to produce probabilities for each class and then takes
  # the loss to be the negative log of the probability of the correct class (we want to maximize the
  # probability, which is equivalent to minimizing the negative log of the probability). Thus, instead
  # of adding a softmax layer to the neural net architecture, we just do the softmax in the loss function.
  # Maybe it's more convenient this way but seems to be the standard.
  criterion = nn.CrossEntropyLoss()
  #optimizer = optim.Adam(net.parameters()) # TODO: somehow training the CNN didn't work when using this; need to figure out why
  optimizer = optim.SGD(net.parameters(), lr=sgd_lr, momentum=0.9)

  if len(sys.argv) > 1 and sys.argv[1] == "ES":
    print "Training CNN with ES..."
    train_net_via_ES(net, trainloader, es_std_dev, es_lr, es_pop_size, criterion, epochs)
  else:
    print "Training CNN with Backprop..."
    train_net_via_backprop(net, trainloader, optimizer, criterion, epochs) # Best: 9827 correct out of 10000 achieved using backprop without maxpooling
  evaluate_net(net, testloader)

