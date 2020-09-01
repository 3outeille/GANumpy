from src.gan_yaae.engine import Node
import numpy as np

class SGD:
    
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def zero_grad(self):
        for W, b in self.params.values():
            W.zero_grad()
            b.zero_grad()

    def step(self):
        for W, b in self.params.values():       
            W.data -= self.lr * W.grad.data
            b.data -= self.lr * b.grad.data

class Adam:
    def __init__(self, params, lr, beta1=0.9, beta2=0.999, epsilon=1e-08):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        self.momentum = {}
        self.rmsprop = {}

        for i, (W, b) in enumerate(self.params.values()):
            self.momentum['vdW' + str(i)] = np.zeros(W.data.shape)
            self.momentum['vdb' + str(i)] = np.zeros(b.data.shape)

            self.rmsprop['sdW' + str(i)] = np.zeros(W.data.shape)
            self.rmsprop['sdb' + str(i)] = np.zeros(b.data.shape)

    def zero_grad(self):
        for W, b in self.params.values():
            W.zero_grad()
            b.zero_grad()

    def step(self):
        for i, (W, b) in enumerate(self.params.values()):
            # Momentum update.
            self.momentum['vdW' + str(i)] = (self.beta1 * self.momentum['vdW' + str(i)]) + (1 - self.beta1) * W.grad.data
            self.momentum['vdb' + str(i)] = (self.beta1 * self.momentum['vdb' + str(i)]) + (1 - self.beta1) * b.grad.data
            # RMSprop update.
            self.rmsprop['sdW' + str(i)] =  (self.beta2 * self.rmsprop['sdW' + str(i)]) + (1 - self.beta2) * (W.grad.data**2)
            self.rmsprop['sdb' + str(i)] =  (self.beta2 * self.rmsprop['sdb' + str(i)]) + (1 - self.beta2) * (b.grad.data**2)

            # Update parameters.
            W.data = W.data - (self.lr * self.momentum['vdW' + str(i)]) / (np.sqrt(self.rmsprop['sdW' + str(i)]) + self.epsilon)
            b.data = b.data - (self.lr * self.momentum['vdb' + str(i)]) / (np.sqrt(self.rmsprop['sdb' + str(i)]) + self.epsilon)

class Linear():
    
    def __init__(self, name, row, column, isLastLayer=False):
        self.name = name
        self.row = row
        self.col = column
        self.isLastLayer = isLastLayer

        self.W = Node(np.random.randn(row, column) * np.sqrt(1./row), requires_grad=True)
        self.b = Node(np.zeros(column), requires_grad=True)

    def __call__(self, X, model=None):
        act = X.matmul(self.W) + self.b
   
        if not self.isLastLayer:
            return act.lrelu()
        elif model == "D":
            return act.sigmoid()
        else:
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
