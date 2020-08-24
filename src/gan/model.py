from src.dcgan.layers import *
from src.dcgan.utils import *
import numpy as np

class Generator():

    def __init__(self):
        self.fc1 = Fc(row=256, column=100)
        self.act1 = LeakyReLU(0.2)
        self.fc2 = Fc(row=512, column=256)
        self.act2 = LeakyReLU(0.2)
        self.fc3 = Fc(row=1024, column=512)
        self.act3 = LeakyReLU(0.2)
        self.fc4 = Fc(row=28*28, column=1024) # 28*28=784=size of MNIST image.
        self.act4 = TanH()

        self.layers = [self.fc1, self.fc2, self.fc3, self.fc4]
    
    def forward(self, X):
        
        out = self.fc1.forward(X)
        out = self.act1.forward(out)
        out = self.fc2.forward(X)
        out = self.act2.forward(out)
        out = self.fc3.forward(X)
        out = self.act3.forward(out)
        out = self.fc4.forward(X)
        out = self.act4.forward(out)
        
        return out

    def backward(self):
        pass
    
    def get_params(self):
        pass

    def set_params(self):
        pass

class Discriminator():

    def __init__(self):
        self.fc1 = Fc(row=1024, column=28*28)
        self.act1 = LeakyReLU(0.2)
        self.fc2 = Fc(row=512, column=1024)
        self.act2 = LeakyReLU(0.2)
        self.fc3 = Fc(row=256, column=512)
        self.act3 = LeakyReLU(0.2)
        self.fc4 = Fc(row=1, column=256)
        self.act4 = Sigmoid()

        self.layers = [self.fc1, self.fc2, self.fc3, self.fc4]
    
    def forward(self, X):
        
        out = self.fc1.forward(X)
        out = self.act1.forward(out)
        out = self.fc2.forward(X)
        out = self.act2.forward(out)
        out = self.fc3.forward(X)
        out = self.act3.forward(out)
        out = self.fc4.forward(X)
        out = self.act4.forward(out)
        
        return out

    def backward(self):
        pass
    
    def get_params(self):
        pass

    def set_params(self):
        pass

class DCGAN():


    def forward(self, X):
        conv1 = self.conv1.forward(X) #(6x28x28)
        act1 = self.tanh1.forward(conv1)
        pool1 = self.pool1.forward(act1) #(6x14x14)

        conv2 = self.conv2.forward(pool1) #(16x10x10)
        act2 = self.tanh2.forward(conv2)
        pool2 = self.pool2.forward(act2) #(16x5x5)
        
        self.pool2_shape = pool2.shape #Need it in backpropagation.
        pool2_flatten = pool2.reshape(self.pool2_shape[0], -1) #(1x400)
    
        fc1 = self.fc1.forward(pool2_flatten) #(1x120)
        act3 = self.tanh3.forward(fc1)
        
        fc2 = self.fc2.forward(act3) #(1x84)
        act4 = self.tanh4.forward(fc2)
        
        fc3 = self.fc3.forward(act4) #(1x10)
    
        y_pred = self.softmax.forward(fc3)

        return y_pred
        
    def backward(self, y_pred, y):
        deltaL = self.softmax.backward(y_pred, y)
        #Compute gradient for weight/bias between fc3 and fc2.
        deltaL, dW5, db5, = self.fc3.backward(deltaL)
        #Compute error at fc2 layer.
        deltaL = self.tanh4.backward(deltaL) #(1x84) 
        
        #Compute gradient for weight/bias between fc2 and fc1.
        deltaL, dW4, db4 = self.fc2.backward(deltaL)
        #Compute error at fc1 layer.
        deltaL = self.tanh3.backward(deltaL) #(1x120)
        
        #Compute gradient for weight/bias between fc1 and pool2 and compute 
        #error too (don't need to backpropagate through tanh here).
        deltaL, dW3, db3 = self.fc1.backward(deltaL) #(1x400)
        deltaL = deltaL.reshape(self.pool2_shape) #(16x5x5)
        
        #Distribute error through pool2 to conv2.
        deltaL = self.pool2.backward(deltaL) #(16x10x10)
        #Distribute error through tanh.
        deltaL = self.tanh2.backward(deltaL)
        
        #Compute gradient for weight/bias at conv2 layer and backpropagate
        #error to conv1 layer.
        deltaL, dW2, db2 = self.conv2.backward(deltaL) #(6x14x14)

        #Distribute error through pool1 by creating a temporary pooling layer
        #of conv1 shape.
        deltaL = self.pool1.backward(deltaL) #(6x28x28)
        #Distribute error through tanh.
        deltaL = self.tanh1.backward(deltaL)
    
        #Compute gradient for weight/bias at conv1 layer and backpropagate
        #error at conv1 layer.
        deltaL, dW1, db1 = self.conv1.backward(deltaL) #(1x32x32)
    
        grads = { 
                'dW1': dW1, 'db1': db1,
                'dW2': dW2, 'db2': db2, 
                'dW3': dW3, 'db3': db3,
                'dW4': dW4, 'db4': db4,
                'dW5': dW5, 'db5': db5
        }

        return grads

    def get_params(self):
        params = {}
        for i, layer in enumerate(self.layers):
            params['W' + str(i+1)] = layer.W['val']
            params['b' + str(i+1)] = layer.b['val']

        return params

    def set_params(self, params):
        for i, layer in enumerate(self.layers):
            layer.W['val'] = params['W'+ str(i+1)]
            layer.b['val'] = params['b' + str(i+1)]