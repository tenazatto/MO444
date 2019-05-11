import numpy as np
import sklearn.metrics as metric


class MLP(object):
    @staticmethod
    def sigmoid(x, derive=False):
        if derive:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def tanh(x, derive=False):
        if derive:
            return 1. - x * x
        return np.tanh(x)

    @staticmethod
    def relu(x, derive=False):
        if derive:
            return 1. * (x > 0)
        return x * (x > 0)

    @staticmethod
    def softmax(x):
        exp_scores = np.exp(x)
        return exp_scores/exp_scores.sum(axis=1, keepdims=True)

    @staticmethod
    def stable_softmax(x):
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps)

    @staticmethod
    def forward(x, w, b, g):
        n = len(w)
        a = [g(np.dot(x, w[0]) + b[0])]
        for i in range(1, n-1):
            a.append(g(np.dot(a[i-1], w[i]) + b[i]))
        a.append(MLP.softmax(np.dot(a[n-2], w[n-1] + b[n-1])))
        return a

    @staticmethod
    def backward(y, a, w, g):
        n = len(w)
        delta = []
        for i in range(0, n):
            delta.append([])

        delta[n-1] = a[n-1] - y
        for i in range(n-2, -1, -1):
            delta[i] = (delta[i+1]).dot(w[i+1].T) * g(a[i], derive=True)

        return delta

    @staticmethod
    def update(x, w, b, a, delta, eta):
        m = x.shape[0]
        c = 1./float(m)
        n = len(w)
        for i in range(n-1, 0, -1):
            w[i] += -eta * c * a[i-1].T.dot(delta[i])
            b[i] += -eta * c * (delta[i]).sum(axis=0)
        w[0] += -eta * c * x.T.dot(delta[0])
        b[0] += -eta * c * (delta[0]).sum(axis=0)

    @staticmethod
    def predict(x, model):
        a = MLP.forward(x, model['w'], model['b'], model['activation'])
        return np.argmax(a[len(a)-1], axis=1)

    @staticmethod
    def build_model(x, y, layers, activation, epsilon, eta, epochs):

        # One-hot coding
        Y = np.zeros((x.shape[0], layers[len(layers)-1]))
        for i in range(x.shape[0]):
            Y[i, y[i]] = 1

        # Set activation function
        g = None
        if activation == 'logistic':
            g = MLP.sigmoid
        elif activation == 'tanh':
            g = MLP.tanh
        elif activation == 'relu':
            g = MLP.relu

        w = []
        b = []
        cost = []
        model = {}
        n_layers = len(layers)

        # Initialize the weights with random numbers
        for i in range(0, n_layers-1):
            w.append(np.random.randn(layers[i], layers[i+1]) * (2 * epsilon) - epsilon)
            b.append(np.random.randn(1, layers[i+1]) * (2 * epsilon) - epsilon)

        # For each epoch
        for epoch in range(epochs):

            # Feed forward
            a = MLP.forward(x, w, b, g)

            # Compute the loss
            loss = -1.0/float(x.shape[0])*np.sum(Y*np.log(a[len(a)-1]) + (1-Y)*np.log(1-a[len(a)-1]))
            cost.append(loss)
            print('Epoch: %d' % (epoch + 1), 'Loss: %f' % loss)

            # Backpropagation
            delta = MLP.backward(Y, a, w, g)

            # Update the weights
            MLP.update(x, w, b, a, delta, eta)

        model['w'] = w
        model['b'] = b
        model['cost'] = cost
        model['activation'] = g

        return model
