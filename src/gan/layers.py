import numpy as np

class Fc():

    def __init__(self, row, column):
        self.row = row
        self.col = column

        # Xavier-Glorot initialization - used for sigmoid, tanh.
        self.W = {'val': np.random.randn(self.row, self.col) * np.sqrt(1./self.col), 'grad': 0}
        # self.b = {'val': np.random.randn(1, self.row) * np.sqrt(1./self.row), 'grad': 0}
        self.b = {'val': np.zeros((1, self.row)), 'grad': 0}
        
        self.cache = None

    def forward(self, fc):
        """
            Performs a forward propagation between 2 fully connected layers.

            Parameters:
            - fc: fully connected layer.
            
            Returns:
            - A_fc: new fully connected layer.
        """
        self.cache = fc
        A_fc = np.dot(fc, self.W['val'].T) + self.b['val']
        return A_fc

    def backward(self, deltaL):
        """
            Returns the error of the current layer and compute gradients.

            Parameters:
            - deltaL: error at last layer.
            
            Returns:
            - new_deltaL: error at current layer.
            - self.W['grad']: weights gradient.
            - self.b['grad']: bias gradient.    
        """
        fc = self.cache
        m = fc.shape[0]

        # Compute gradient.
        # self.W['grad'] = (1/m) * np.dot(deltaL.T, fc)
        # self.b['grad'] = (1/m) * np.sum(deltaL, axis = 0)
        
        self.W['grad'] = np.dot(deltaL.T, fc)
        self.b['grad'] = np.sum(deltaL, axis = 0)
        
        #Compute error.
        new_deltaL = np.dot(deltaL, self.W['val']) 

        #We still need to multiply new_deltaL by the derivative of the activation.
        return new_deltaL, self.W['grad'], self.b['grad']
    
class SGD():

    def __init__(self, params):
        self.params = params

    def update_params(self, grads, lr):
        for key in self.params:
            self.params[key] = self.params[key] - lr * grads['d' + key]
        return self.params        

class AdamGD():

    def __init__(self, lr, beta1, beta2, epsilon, params):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.params = params
        
        self.momentum = {}
        self.rmsprop = {}

        for key in self.params:
            self.momentum['vd' + key] = np.zeros(self.params[key].shape)
            self.rmsprop['sd' + key] = np.zeros(self.params[key].shape)

    def update_params(self, grads):
        
        for key in self.params:
            # Momentum update.
            self.momentum['vd' + key] = (self.beta1 * self.momentum['vd' + key]) + (1 - self.beta1) * grads['d' + key] 
            # RMSprop update.
            self.rmsprop['sd' + key] =  (self.beta2 * self.rmsprop['sd' + key]) + (1 - self.beta2) * (grads['d' + key]**2)
            # Update parameters.
            self.params[key] = self.params[key] - (self.lr * self.momentum['vd' + key]) / (np.sqrt(self.rmsprop['sd' + key]) + self.epsilon)  

        return self.params

class TanH():
 
    def __init__(self):
        self.cache = None

    def forward(self, X):
        """
            Apply tanh function to X.

            Parameters:
            - X: input tensor.
        """
        self.cache = X
        return np.tanh(X)

    def backward(self, new_deltaL):
        """
            Finishes computation of error by multiplying new_deltaL by the
            derivative of tanH.

            Parameters:
            - new_deltaL: error previously computed.
        """
        X = self.cache
        return new_deltaL * (1. - np.tanh(X)**2)

class LeakyReLU():

    def __init__(self, alpha):
        self.alpha = alpha
        self.cache = None

    def forward(self, X):
        self.cache = X
        return np.where(X > 0, X, X * self.alpha)  

    def backward(self, new_deltaL):
        X = self.cache
        dX = np.ones_like(X)
        dX[X < 0] = self.alpha
        return dX * new_deltaL

class Sigmoid():
    
    def __init__(self):
        self.cache = None

    def forward(self, X):
        self.cache = X
        return 1. / (1. + np.exp(-X))

    def backward(self, new_deltaL):
        X = self.cache
        sigmoid = 1. / (1. + np.exp(-X))
        return new_deltaL * (sigmoid * (1. - sigmoid))

class BinaryCrossEntropyLoss():

    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon
    
    def get(self, y_pred, y):
        """
            Return the negative log likelihood and the error at the last layer.
            
            Parameters:
            - y_pred: model predictions.
            - y: ground truth labels.
        """
        loss = -np.mean((y * np.log(y_pred + self.epsilon) + (1. - y) * np.log(1. - y_pred + self.epsilon)))
        return loss
