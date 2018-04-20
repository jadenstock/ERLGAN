import torch
from torch.autograd import Variable
from torch.autograd import Function

# NOTE: It might be a good idea to make this False explicitly to avoid confusion
# Essentially, requires_grad=False means "treat this like a constant and don't
# backprop gradient through this"
# https://discuss.pytorch.org/t/operation-between-tensor-and-variable/1286
print "===== Basics of Tensor, Variable, Grad ====="
A = Variable(torch.eye(2), requires_grad=False) # identity
x = Variable(torch.randn(2), requires_grad=True) # N(0,1) entries
b = Variable(torch.rand(2), requires_grad=False) # unif[0,1) entries

print "Constant Matrix A: {}".format(A)
print "Initial x: {}".format(x)
print "Constant Vector: {}".format(b)

# NOTE: this works because both arguments are Variable
# the only thing not allowed is calling matmul on a regular Tensor with a Variable
y = torch.matmul(A, x) - b # requires_grad=True here because it is a function of a Variable with requires_grad=True
print "Variable y = Ax - b: {}".format(y)

loss = 0.5 * y.norm().pow(2)
print "Loss (1/2) * ||Ax - b||^2: {}".format(loss)
loss.backward() # NOTE: can only call backward on a scalar (which makes sense)

print "Gradient of loss w.r.t. x: {}".format(x.grad)
# NOTE: This is None because it is an intermediate node (Variable) in the computation and hence,
# its gradient isn't stored for memory purposes:
# https://discuss.pytorch.org/t/why-cant-i-see-grad-of-an-intermediate-variable/94
print "Gradient of loss w.r.t. y: {}".format(y.grad)

g = torch.matmul(torch.t(A), torch.matmul(A, x) - b).data
# NOTE: this is the same as x.grad as desired
print "Manually Calculated Gradient of loss w.r.t. x: {}".format(g)

print "===== Run Least Squares via Gradient Descent ====="
tol = 1e-10
lr = 0.2
iters = 0
max_iters = 100
while loss.data[0] > tol and iters < max_iters:
	# NOTE: zeroing the gradient is crucial as if you leave them there, they
	# accumulate with the new gradients so you won't converge
	x.grad.data.zero_()
	loss = 0.5 * (torch.matmul(A, x) - b).norm().pow(2)
	print "Loss at iteration {}: {}".format(iters, loss.data[0])
	loss.backward()
#	print "Gradient at Iteration {}: {}".format(iters, x.grad.data)
	x.data.sub_(lr * x.grad.data)
	iters += 1

print "Converged in {} Iterations".format(iters)
print "Final x: {}".format(x)
print "Final Loss: {}".format(0.5 * (torch.matmul(A, x) - b).norm().pow(2).data[0])
truth = torch.matmul(torch.inverse(A.data), b.data)
print "Truth: {}".format(truth)
print "Truth Loss: {}".format(0.5 * (torch.matmul(A.data, truth) - b.data).norm() ** 2)

