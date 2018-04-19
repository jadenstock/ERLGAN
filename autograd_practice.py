import torch
from torch.autograd import Variable
from torch.autograd import Function

"""
# Aha: One cannot implement least squares this way because
# it mixes Tensors and Variable, which isn't allowed. The right way
# to do this is to wrap A and b in Variable. Now, we can ensure that
# they behave like regular constants in the sense that they don't
# receive gradients when backpropagating by making "requires_grad"
# False, which is the default.
# https://discuss.pytorch.org/t/operation-between-tensor-and-variable/1286

A = torch.eye(2) # constant matrix (data matrix; features)
x = Variable(torch.randn(2), requires_grad=True) # vector variable
b = torch.randn(2) # constant vector (labels)
print "Variable x: {}".format(x)

y = torch.matmul(A, x) - b
print "Variable y = Ax - b: {}".format(y)

# Here is the right way to implement.
"""
# NOTE: It might be a good idea to make this false explicitly to avoid confusion
A = Variable(torch.rand(2, 2), requires_grad=False) # matrix variable that is constant (basically a regular tensor)
x = Variable(torch.randn(2), requires_grad=True) # vector variable
b = Variable(torch.randn(2), requires_grad=False) # vector variable that is constant (basically a regular vector)

print "Constant Matrix A: {}".format(A)
print "Variable x: {}".format(x)
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
print "Manually Calculated Gradient of loss w.r.t. x: {}".format(g) # NOTE: this is the same as x.grad as desired

