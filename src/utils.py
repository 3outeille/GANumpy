import os
import gzip
import math
import pickle
import imageio
import itertools
import numpy as np
import urllib.request
from PIL import Image
from skimage import transform
import matplotlib.pyplot as plt
import concurrent.futures as cf
from src.data import *
from src.engine import Node

def download_mnist(filename):
    """
        Downloads dataset from filename.

        Parameters:
        - filename: [
                        ["training_images","train-images-idx3-ubyte.gz"],
                        ["test_images","t10k-images-idx3-ubyte.gz"],
                        ["training_labels","train-labels-idx1-ubyte.gz"],
                        ["test_labels","t10k-labels-idx1-ubyte.gz"]
                    ]
    """
    # Make data/ accessible from every folders.
    terminal_path = ['src/data/', 'data/']
    dirPath = None
    for path in terminal_path:
        if os.path.isdir(path):
            dirPath = path
    if dirPath == None:
        raise FileNotFoundError("extract_mnist(): Impossible to find data/ from current folder. You need to manually add the path to it in the \'terminal_path\' list and the run the function again.")

    base_url = "http://yann.lecun.com/exdb/mnist/"
    for elt in filename:
        print("Downloading " + elt[1] + " in data/ ...")
        urllib.request.urlretrieve(base_url + elt[1], dirPath + elt[1])
    print("Download complete.")

def extract_mnist(filename):
    """
        Extracts dataset from filename.

        Parameters:
        - filename: [
                        ["training_images","train-images-idx3-ubyte.gz"],
                        ["test_images","t10k-images-idx3-ubyte.gz"],
                        ["training_labels","train-labels-idx1-ubyte.gz"],
                        ["test_labels","t10k-labels-idx1-ubyte.gz"]
                    ]
    """
    # Make data/ accessible from every folders.
    terminal_path = ['src/data/', 'data/']
    dirPath = None
    for path in terminal_path:
        if os.path.isdir(path):
            dirPath = path
    if dirPath == None:
        raise FileNotFoundError("extract_mnist(): Impossible to find data/ from current folder. You need to manually add the path to it in the \'terminal_path\' list and the run the function again.")

    mnist = {}
    for elt in filename[:2]:
        print('Extracting data/' + elt[0] + '...')
        with gzip.open(dirPath + elt[1]) as f:
            #According to the doc on MNIST website, offset for image starts at 16.
            mnist[elt[0]] = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(-1, 1, 28, 28)
    
    for elt in filename[2:]:
        print('Extracting data/' + elt[0] + '...')
        with gzip.open(dirPath + elt[1]) as f:
            #According to the doc on MNIST website, offset for label starts at 8.
            mnist[elt[0]] = np.frombuffer(f.read(), dtype=np.uint8, offset=8)

    print('Files extraction: OK') 

    return mnist

def load(filename, numbers=[]):
    """
        Loads dataset to variables.

        Parameters:
        - filename: [
                        ["training_images","train-images-idx3-ubyte.gz"],
                        ["test_images","t10k-images-idx3-ubyte.gz"],
                        ["training_labels","train-labels-idx1-ubyte.gz"],
                        ["test_labels","t10k-labels-idx1-ubyte.gz"]
                  ]
    """
    # Make data/ accessible from every folders.
    terminal_path = ['src/data/', 'data/']
    dirPath = None
    for path in terminal_path:
        if os.path.isdir(path):
            dirPath = path
    if dirPath == None:
        raise FileNotFoundError("extract_mnist(): Impossible to find data/ from current folder. You need to manually add the path to it in the \'terminal_path\' list and the run the function again.")

    L = [elt[1] for elt in filename]   
    count = 0 

    #Check if the 4 .gz files exist.
    for elt in L:
        if os.path.isfile(dirPath + elt):
            count += 1

    #If the 4 .gz are not in data/, we download and extract them.
    if count != 4:
        download_mnist(filename)
        mnist = extract_mnist(filename)
    else: #We just extract them.
        mnist = extract_mnist(filename)

    print('Loading dataset: OK')
    
    if numbers != []:
        newtrainX = []
        for idx in range(0,len(mnist["training_images"])):
            if mnist["training_labels"][idx] in numbers:
                newtrainX.append(mnist["training_images"][idx, :])
        return np.array(newtrainX)
    else:
        return mnist["training_images"]

def dataloader(X, BATCH_SIZE):
    """
        Returns a data generator.

        Parameters:
        - X: dataset examples.
        - y: ground truth labels.
    """
    n = len(X)
    for t in range(0, n, BATCH_SIZE):
        yield X[t:t+BATCH_SIZE, ...]

def show_result(G, epoch, path, show=False, save=False):
    fixed_noise = np.random.randn(5*5, 100)
    fixed_noise = Node(fixed_noise, requires_grad=False)
    test_images = G(fixed_noise)

    size_figure_grid = 5

    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(5*5):
        i = k // 5
        j = k % 5
        ax[i, j].cla()
        ax[i, j].imshow(test_images.data[k, :].reshape((28, 28)), cmap='gray')

    label = 'Epoch {0}'.format(epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

def show_train_hist(hist, path, show=False, save=False):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()