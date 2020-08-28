from src.gan.layers import *
from src.gan.utils import *
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
        out = self.fc2.forward(out)
        out = self.act2.forward(out)
        out = self.fc3.forward(out)
        out = self.act3.forward(out)
        out = self.fc4.forward(out)
        out = self.act4.forward(out)
        
        return out

    def backward(self, deltaL):        
        #######################################
        #		Fake images gradients.
        #		-log(D(G(z)))
        #######################################

        deltaL = self.act4.backward(deltaL)
        deltaL, dW4, db4 = self.fc4.backward(deltaL)
        
        deltaL = self.act3.backward(deltaL)
        deltaL, dW3, db3 = self.fc3.backward(deltaL)
        
        deltaL = self.act2.backward(deltaL)
        deltaL, dW2, db2 = self.fc2.backward(deltaL)
        
        deltaL = self.act1.backward(deltaL)
        deltaL, dW1, db1 = self.fc1.backward(deltaL)

        grads = { 
            'dW1': dW1 , 'db1': db1,
            'dW2': dW2 , 'db2': db2, 
            'dW3': dW3 , 'db3': db3,
            'dW4': dW4 , 'db4': db4
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
        out = self.fc2.forward(out)
        out = self.act2.forward(out)
        out = self.fc3.forward(out)
        out = self.act3.forward(out)
        out = self.fc4.forward(out)
        out = self.act4.forward(out)

        return out

    def backward(self, real_deltaL, fake_deltaL):
        #######################################
        #		Real images gradients.
        #		-log(D(x))
        #######################################

        real_deltaL = self.act4.backward(real_deltaL)
        real_deltaL, real_dW4, real_db4 = self.fc4.backward(real_deltaL)

        real_deltaL = self.act3.backward(real_deltaL)
        real_deltaL, real_dW3, real_db3 = self.fc3.backward(real_deltaL)

        real_deltaL = self.act2.backward(real_deltaL)
        real_deltaL, real_dW2, real_db2 = self.fc2.backward(real_deltaL)

        real_deltaL = self.act1.backward(real_deltaL)
        real_deltaL, real_dW1, real_db1 = self.fc1.backward(real_deltaL)

        #######################################
        #		Fake images gradients.
        #		-log(1 - D(G(z)))
        #######################################

        fake_deltaL = self.act4.backward(fake_deltaL)
        fake_deltaL, fake_dW4, fake_db4 = self.fc4.backward(fake_deltaL)

        fake_deltaL = self.act3.backward(fake_deltaL)
        fake_deltaL, fake_dW3, fake_db3 = self.fc3.backward(fake_deltaL)

        fake_deltaL = self.act2.backward(fake_deltaL)
        fake_deltaL, fake_dW2, fake_db2 = self.fc2.backward(fake_deltaL)

        fake_deltaL = self.act1.backward(fake_deltaL)
        fake_deltaL, fake_dW1, fake_db1 = self.fc1.backward(fake_deltaL)

        grads = { 
                'dW1': real_dW1 + fake_dW1, 'db1': real_db1 + fake_db1,
                'dW2': real_dW2 + fake_dW2, 'db2': real_db2 + fake_db2, 
                'dW3': real_dW3 + fake_dW3, 'db3': real_db3 + fake_db3,
                'dW4': real_dW4 + fake_dW4, 'db4': real_db4 + fake_db4
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