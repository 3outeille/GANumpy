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
        out = self.fc2.forward(X)
        out = self.act2.forward(out)
        out = self.fc3.forward(X)
        out = self.act3.forward(out)
        out = self.fc4.forward(X)
        out = self.act4.forward(out)
        
        return out

    def backward(self):
        self.act4.backward()
        self.fc4.backward()

        self.act3.backward()
        self.fc3.backward()
        
        self.act2.backward()
        self.fc2.backward()
        
        self.act1.backward()
        self.fc1.backward()        

    
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
        deltaL = self.softmax.backward(y_pred, y)
    
    def get_params(self):
        pass

    def set_params(self):
        pass