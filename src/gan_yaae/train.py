from src.gan_yaae.engine import Node
from src.gan_yaae.gan import Generator, Discriminator, Adam, SGD
from src.gan_yaae.gan_utils import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from timeit import default_timer as timer
import os
import pickle

filename = [
        ["training_images","train-images-idx3-ubyte.gz"],
        ["test_images","t10k-images-idx3-ubyte.gz"],
        ["training_labels","train-labels-idx1-ubyte.gz"],
        ["test_labels","t10k-labels-idx1-ubyte.gz"]
]

def train():
    NB_EPOCH = 300
    BATCH_SIZE = 1
    LR = 0.0002

    # results save folder
    if not os.path.isdir('MNIST_GAN_results'):
        os.mkdir('MNIST_GAN_results')
    if not os.path.isdir('MNIST_GAN_results/training_results'):
        os.mkdir('MNIST_GAN_results/training_results')
    if not os.path.isdir('MNIST_GAN_results/save_weights'):
        os.mkdir('MNIST_GAN_results/save_weights')

    def binary_cross_entropy_loss(y_pred, labels):
        samples = y_pred.shape[0]
        # 1. Compute binary cross entropy loss.
        # Add + 1e-8 to avoid division by zero.
        loss = labels * ((y_pred + 1e-8).log()) + (1. - labels) * ((1. + 1e-8 - y_pred).log())
        train_loss = loss.sum(keepdims=False) / -samples

        return train_loss

    criterion = binary_cross_entropy_loss

    def real_loss(D_out, smooth=False):
        batch_size = D_out.shape[0]
        if smooth:
            labels = np.ones(batch_size) * 0.9
        else:
            labels = np.ones(batch_size) # real labels = 1.
        labels = Node(labels.reshape((batch_size, 1)), requires_grad=False)
        loss = criterion(D_out, labels)
        return loss

    def fake_loss(D_out):
        batch_size = D_out.shape[0]
        labels = np.zeros(batch_size) # fake labels = 0.    
        labels = Node(labels.reshape((batch_size, 1)), requires_grad=False)
        loss = criterion(D_out, labels)
        return loss

    print("\n----------------EXTRACTION---------------\n")
    X, _, _, _ = load(filename)
    X = X[:BATCH_SIZE, ...]

    print("\n--------------PREPROCESSING--------------\n")
    X = (X - 127.5) / 127.5
    print("Normalize dataset: OK")
    
    G = Generator(nin=100, nouts=[256,512,1024,28*28])
    D = Discriminator(nin=28*28, nouts=[1024,512,256,1])
    # G = Generator(nin=100, nouts=[128,28*28])
    # D = Discriminator(nin=28*28, nouts=[128,1])

    G_optimizer = Adam(params=G.parameters(), lr=LR, beta1=0.5)
    D_optimizer = Adam(params=D.parameters(), lr=LR, beta2=0.5)
    
    print("----------------TRAINING-----------------\n")
    
    print("EPOCHS: {}".format(NB_EPOCH))
    print("BATCH_SIZE: {}".format(BATCH_SIZE))
    print("LR: {}".format(LR))
    print()

    nb_examples = len(X)
    
    train_hist = {}
    train_hist['D_losses'] = []
    train_hist['G_losses'] = []


    for epoch in range(NB_EPOCH):
        
        pbar = trange(nb_examples // BATCH_SIZE)
        train_loader = dataloader(X, BATCH_SIZE)
        
        D_losses, G_losses = [], []
        start = timer()

        for i, real_images in zip(pbar, train_loader):

            # ---------TRAIN THE DISCRIMINATOR ----------------
            D_optimizer.zero_grad()
                        
            # # 1. Train with real images.
            real_images = real_images.reshape((BATCH_SIZE, -1))
            real_images = Node(real_images, requires_grad=False)
            # Compute the discriminator losses on real images
            # smooth the real labels.
            D_real = D(real_images)
            D_real_loss = real_loss(D_real, smooth=True)

            # 2. Train with fake images.
            # Generate fake images.
            z = Node(np.random.randn(BATCH_SIZE, 100), requires_grad=False)
            fake_images = G(z)

            # 3. Compute the discriminator losses on fake images.
            D_fake = D(fake_images)
            D_fake_loss = fake_loss(D_fake)

            # 4. Add up real and fake loss.
            D_loss = D_real_loss + D_fake_loss

            # 5. Perform backprop and optimization step.
            D_loss.backward()
            D_optimizer.step()

            D_losses.append(D_loss.data)

            # ---------TRAIN THE GENERATOR ----------------
            G_optimizer.zero_grad()

            # 1. Generate fake images.
            z = Node(np.random.randn(BATCH_SIZE, 100), requires_grad=False)
            fake_images = G(z)
            
            # 2. Compute the discriminator loss on fake images
            # using flipped labels.
            D_fake = D(fake_images)
            G_loss = real_loss(D_fake) # use real loss to flip labels.
          
            # 3. Perform backprop and optimization step.        
            G_loss.backward()
            G_optimizer.step()

            G_losses.append(G_loss.data)

        end = timer()
        # Print discriminator and generator loss.
        info = "[Epoch {}/{}] ({:0.3f}s): D_loss = {:0.6f} | G_loss = {:0.6f}"
        print(info.format(epoch+1, NB_EPOCH, end-start, np.mean(D_losses), np.mean(G_losses)))

        # Visualize generator learning.
        path = 'MNIST_GAN_results/training_results/MNIST_GAN_' + str(epoch + 1) + '.png'
        show_result(G, epoch+1, save=True, path=path)

        train_hist['D_losses'].append(np.mean(D_losses))
        train_hist['G_losses'].append(np.mean(G_losses))

    pbar.close()
    print("Training finish!... save training results")

    with open('MNIST_GAN_results/train_hist.pkl', 'wb') as f:
        pickle.dump(train_hist, f)

    show_train_hist(train_hist, save=True, path='MNIST_GAN_results/MNIST_GAN_train_hist.png')
    
    # Create Gif
    images = []
    for e in range(NB_EPOCH):
        img_name = 'MNIST_GAN_results/training_results/MNIST_GAN_' + str(e + 1) + '.png'
        images.append(imageio.imread(img_name))
    imageio.mimsave('MNIST_GAN_results/generation_animation.gif', images, fps=5)

train()
