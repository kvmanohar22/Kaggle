import numpy as np
import theano
import theano.tensor as T
from theano import function, shared
rng = np.random

#number of training examples
N = 400
features = 784

#generating the dataset (input, target_Class)
D = (rng.randn(N, features), rng.randint(size=N, low=0, high=2))
training_steps = 1000

x = T.dmatrix("x")
y = T.dvector("y")

#initialise the weights and bias term
w = shared(rng.randn(features), name="w")
b = shared(0., name="b")

#declare the hypothesis sigmoid function
h = 1 / (1 + T.exp(-T.dot(x, w) - b))
pred = h > 0.5

#declare the cost function
J = -y * T.log(h) - (1-y) * T.log(1-h)
cost = J.mean() + 0.01 * (w ** 2).sum()
grad_w, grad_b = T.grad(cost, [w, b])

#create functions
train = function(inputs=[x, y], outputs=[pred, cost], updates=((w, w-0.1*grad_w), (b, b-0.1*grad_b)))

predict = function(inputs=[x], outputs=pred)

#train the model
for i in xrange(training_steps):
	pred, cost = train(D[0], D[1])
	print 'Iteration: ', i+1, '\tCost: ', cost

print 'Final Model:\n'
print 'Comparing target values and prediction values over training set'
print 'target values: '
print D[1]
print 'prediction of training set: '
print predict(D[0])
print 'Done!'

