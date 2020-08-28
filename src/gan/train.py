from src.gan.utils import *
from src.gan.layers import *
from src.gan.model import Generator, Discriminator
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from timeit import default_timer as timer

filename = [
        ["training_images","train-images-idx3-ubyte.gz"],
        ["test_images","t10k-images-idx3-ubyte.gz"],
        ["training_labels","train-labels-idx1-ubyte.gz"],
        ["test_labels","t10k-labels-idx1-ubyte.gz"]
]

def train():
    NB_EPOCH = 10
    BATCH_SIZE = 100
    LR = 0.0002
    epsilon = 10e-8

    # results save folder
    if not os.path.isdir('MNIST_GAN_results'):
        os.mkdir('MNIST_GAN_results')
    if not os.path.isdir('MNIST_GAN_results/training_results'):
        os.mkdir('MNIST_GAN_results/training_results')
    if not os.path.isdir('MNIST_GAN_results/save_weights'):
        os.mkdir('MNIST_GAN_results/save_weights')

    criterion = BinaryCrossEntropyLoss()

    def real_loss(D_out, smooth=False):
        batch_size = D_out.shape[0]
        if smooth:
            labels = np.ones(batch_size) * 0.9
        else:
            labels = np.ones(batch_size) # real labels = 1.

        loss = criterion.get(D_out.squeeze(), labels)
        deltaL = -1./(D_out + epsilon)
        return loss, deltaL

    def fake_loss(D_out):
        batch_size = D_out.shape[0]
        labels = np.zeros(batch_size) # fake labels = 0.    
        loss = criterion.get(D_out.squeeze(), labels)
        deltaL = -1./(D_out + epsilon)
        return loss, deltaL

    print("\n----------------EXTRACTION---------------\n")
    X, y, X_test, y_test = load(filename)
    X = X[:100, ...]

    print("\n--------------PREPROCESSING--------------\n")
    X = (X - 0.5) / 0.5
    print("Normalize dataset: OK")
    
    G = Generator()
    D = Discriminator()

    G_optimizer = AdamGD(lr = LR, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, params = G.get_params())
    D_optimizer = AdamGD(lr = LR, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, params = D.get_params())
    
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
        train_loader = dataloader(X, y, BATCH_SIZE)
        
        D_losses, G_losses = [], []
        start = timer()

        for i, (real_images, _) in zip(pbar, train_loader):
            
            # ---------TRAIN THE DISCRIMINATOR ----------------
                        
            # # 1. Train with real images.
            real_images = real_images.reshape((BATCH_SIZE, -1))
            # Compute the discriminator losses on real images
            # smooth the real labels.
            D_real = D.forward(real_images)
            D_real_loss, real_deltaL = real_loss(D_real, smooth=True)

            # 2. Train with fake images.
            # Generate fake images.
            z = np.random.randn(BATCH_SIZE, 100)
            fake_images = G.forward(z)

            # 3. Compute the discriminator losses on fake images.
            D_fake = D.forward(fake_images)
            D_fake_loss, fake_deltaL = fake_loss(D_fake)

            # 4. Add up real and fake loss.
            D_loss = D_real_loss + D_fake_loss

            # 5. Perform backprop and optimization step.
            grads = D.backward(real_deltaL, fake_deltaL)
            params = D_optimizer.update_params(grads)
            D.set_params(params)

            D_losses.append(D_loss.item())

            # ---------TRAIN THE GENERATOR ----------------

            # 1. Generate fake images.
            z = np.random.randn(BATCH_SIZE, 100)
            fake_images = G.forward(z)
    
            # 2. Compute the discriminator loss on fake images
            # using flipped labels.
            D_fake = D.forward(fake_images)
            G_loss, deltaL = real_loss(D_fake) # use real loss to flip labels.
          
            # 3. Perform backprop and optimization step.        
            grads = G.backward(deltaL)
            params = G_optimizer.update_params(grads)
            G.set_params(params)

            G_losses.append(G_loss.item())

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
    save_params_to_file(G, "generator_param.pkl")
    save_params_to_file(D, "discriminator_param.pkl")

    with open('MNIST_GAN_results/train_hist.pkl', 'wb') as f:
        pickle.dump(train_hist, f)

    show_train_hist(train_hist, save=True, path='MNIST_GAN_results/MNIST_GAN_train_hist.png')

    # Create Gif
    images = []
    for e in range(NB_EPOCH):
        img_name = 'MNIST_GAN_results/training_results/MNIST_GAN_' + str(e + 1) + '.png'
        images.append(imageio.imread(img_name))
    imageio.mimsave('MNIST_GAN_results/generation_animation.gif', images, fps=5)

# Uncomment to launch training.
train()
