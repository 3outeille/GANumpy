from src.dcgan.utils import *
from src.dcgan.layers import *
from src.dcgan.model import Generator, Discriminator
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import trange
from timeit import default_timer as timer

filename = [
        ["training_images","train-images-idx3-ubyte.gz"],
        ["test_images","t10k-images-idx3-ubyte.gz"],
        ["training_labels","train-labels-idx1-ubyte.gz"],
        ["test_labels","t10k-labels-idx1-ubyte.gz"]
]

def real_loss():
    pass

def fake_loss():
    pass

def train():
    NB_EPOCH = 1
    BATCH_SIZE = 100
    LR = 0.0002

    print("\n----------------EXTRACTION---------------\n")
    X, y, X_test, y_test = load(filename)
    
    print("\n--------------PREPROCESSING--------------\n")
    X = (X - 0.5) / 0.5
    print("Normalize dataset: OK")
    
    G = Generator()
    D = Discriminator()

    D_optimizer = AdamGD(lr = LR, beta1 = 0.5, beta2 = 0.999, epsilon = 1e-8, params = D.get_params())
    G_optimizer = AdamGD(lr = LR, beta1 = 0.5, beta2 = 0.999, epsilon = 1e-8, params = G.get_params())
    
    print("----------------TRAINING-----------------\n")

    print("EPOCHS: {}".format(NB_EPOCH))
    print("BATCH_SIZE: {}".format(BATCH_SIZE))
    print("LR: {}".format(lr))
    print()

    nb_train_examples = len(X_train)
    losses = []

    for epoch in range(NB_EPOCH):
        
        pbar = trange(nb_train_examples // BATCH_SIZE)
        train_loader = dataloader(X_train, y_train, BATCH_SIZE)

        start = timer()

        for i, (real_images, _) in zip(pbar, train_loader):
            
            # ---------TRAIN THE DISCRIMINATOR ----------------
                        
            # 1. Train with real images.

            # Compute the discriminator losses on real images
            # smooth the real labels.
            D_real = D.forward(real_images)
            D_real_loss = real_loss(D_real, smooth=True)

            # 2. Train with fake images.
            # Generate fake images.
            z = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100, 1, 1))
            fake_images = G.forward(z)

            # 3. Compute the discriminator losses on fake images.
            D_fake = D.forward(fake_images)
            D_fake_loss = fake_loss(D_fake)

            # 4. Add up real and fake loss.
            D_loss = D_real_loss + D_fake_loss

            # 5. Perform backprop and optimization step.
            grads = D.backward()
            params = D_optimizer.update_params(grads)
            D.set_params(params)

            # ---------TRAIN THE GENERATOR ----------------

            # 1. Generate fake images.
            z = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100, 1, 1))
            fake_images = G.forward(z)

            # 2. Compute the discriminator loss on fake images
            # using flipped labels.
            D_fake = D.forward(fake_images)
            G_loss = real_loss(D_fake) # use real loss to flip labels.

            # 3. Perform backprop and optimization step.        
            grads = G.backward()
            params = G_optimizer.update_params(grads)
            G.set_params(params)G_optimizer.step()

            #pbar.set_description("[Train] Epoch {}".format(epoch+1))
        
        end = timer()

        # Append discriminator loss and generator loss.
        losses.append(D_loss, G_loss)
        # Print discriminator and generator loss.
        info = "[Epoch {}/{}] ({:0.3f}s}): D_loss = {:0.6f} | G_loss = {:0.6f}"
        print(info.format(epoch+1, EPOCHS, end-start, D_loss, G_loss))

    pbar.close()

    # fig = plt.figure(figsize=(10,10))
    # fig.add_subplot(2, 1, 1)

    # plt.plot(train_costs)
    # plt.title("Training loss")
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')

    # fig.add_subplot(2, 1, 2)
    # plt.plot(val_costs)
    # plt.title("Validation loss")
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')

    # plt.show()

# Uncomment to launch training.
# train()