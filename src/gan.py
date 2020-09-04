import torch
import numpy as np
from src.engine import Node

class Adam:
    def __init__(self, params, lr, beta1=0.9, beta2=0.999, epsilon=1e-08):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        self.m = {}
        self.v = {}

        for i, (W, b) in enumerate(self.params.values()):
            self.m['W' + str(i)] = np.zeros(W.data.shape)
            self.m['b' + str(i)] = np.zeros(b.data.shape)
            self.v['W' + str(i)] = np.zeros(W.data.shape)
            self.v['b' + str(i)] = np.zeros(b.data.shape)

    def zero_grad(self):
        for W, b in self.params.values():
            W.zero_grad()
            b.zero_grad()

    def step(self):
        for i, (W, b) in enumerate(self.params.values()):

            self.m['W' + str(i)] = (self.beta1 * self.m['W' + str(i)]) + (1 - self.beta1) * W.grad.data
            self.m['b' + str(i)] = (self.beta1 * self.m['b' + str(i)]) + (1 - self.beta1) * b.grad.data

            self.v['W' + str(i)] =  (self.beta2 * self.v['W' + str(i)]) + (1 - self.beta2) * (W.grad.data**2)
            self.v['b' + str(i)] =  (self.beta2 * self.v['b' + str(i)]) + (1 - self.beta2) * (b.grad.data**2)

            mW_hat = self.m['W' + str(i)] / (1 - self.beta1)
            mb_hat = self.m['b' + str(i)] / (1 - self.beta1)
            
            vW_hat = self.v['W' + str(i)] / (1 - self.beta2)
            vb_hat = self.v['b' + str(i)] / (1 - self.beta2)

            # Update parameters.
            W.data = W.data - (self.lr * mW_hat) / (np.sqrt(vW_hat) + self.epsilon)
            b.data = b.data - (self.lr * mb_hat) / (np.sqrt(vb_hat) + self.epsilon)

class Linear():
    
    def __init__(self, name, row, column, isLastLayer=False):
        self.name = name
        self.row = row
        self.col = column
        self.isLastLayer = isLastLayer

        #He Normal initialization.
        scaleW = np.sqrt(2. / (row + column))
        scaleb = np.sqrt(2. / (1 + column))
        self.W = Node(np.random.uniform(-scaleW, scaleW, (row, column)), requires_grad=True)
        self.b = Node(np.random.uniform(-scaleb, scaleb, (1, column)), requires_grad=True)
       
    def __call__(self, X, model=None):
        act = X.matmul(self.W) + self.b

        if not self.isLastLayer:
            act = act.lrelu(0.2)
            act = act.dropout(0.3)
            return act
        elif model == "D":
            return act.sigmoid()
        elif model == "G":
            return act.tanh()

    def __repr__(self):
        return f"({self.name}): Linear(row={self.row}, column={self.col}, isLastLayer={self.isLastLayer})\n"
        
class Generator:

    def __init__(self, nin, nouts):
        sizes = [nin] + nouts
        self.layers = [Linear(f'Linear{i}', sizes[i], sizes[i+1], isLastLayer=(i == len(nouts)-1)) for i in range(len(nouts))]

    def __call__(self, X):
        out = X
        for layer in self.layers:
            out = layer(out, model="G")
        return out

    def __repr__(self):
        s = "model(\n"
        for layer in self.layers:
            s += "   " + str(layer)
        s += ")"
        return s

    def parameters(self):
        params = {}
        for i, layer in enumerate(self.layers):
            params[f'Linear{i}'] = layer.W, layer.b
        return params

class Discriminator:

    def __init__(self, nin, nouts):
        sizes = [nin] + nouts
        self.layers = [Linear(f'Linear{i}', sizes[i], sizes[i+1], isLastLayer=(i == len(nouts)-1)) for i in range(len(nouts))]

    def __call__(self, X):
        out = X
        for layer in self.layers:
            out = layer(out, model="D")
        return out

    def __repr__(self):
        s = "model(\n"
        for layer in self.layers:
            s += "   " + str(layer)
        s += ")"
        return s
        
    def parameters(self):
        params = {}
        for i, layer in enumerate(self.layers):
            params[f'Linear{i}'] = layer.W, layer.b
        return params