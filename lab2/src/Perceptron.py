import numpy as np

class Perceptron():
    def __init__(self, eta):
        self.eta = eta

    def activation(self, W, inputs):
        return np.dot(inputs, W)

    def threshold(self, outputs):
        return np.where(outputs > 0, 1, 0)

    def update_weights(self, inputs, labels, W ):
        predictions = self.activation(W, inputs)
        T = self.threshold(predictions)
        e = T - labels
        return -self.eta*np.dot(e.T, inputs), e

    def compute_cost(self, e):
        return np.mean(e**2) # mse
        # return np.where((e) == 0, 0, 1).mean()

    def train(self,inputs, labels, W, epochs):
        cost = []
        WW = W.copy()
        for epoch in range(epochs):
            dW, e = self.update_weights(inputs, labels, WW)
            c = self.compute_cost(e)
            cost.append(c)
            WW += dW.T
        return WW, cost
