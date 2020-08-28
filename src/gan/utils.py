import os
import gzip
import math
import pickle
import imageio
import itertools
import numpy as np
import urllib.request
from PIL import Image
from src.gan.data import *
from skimage import transform
import matplotlib.pyplot as plt
import concurrent.futures as cf

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
    terminal_path = ['src/gan/data/', 'gan/data/', '../gan/data/', 'data/', '../data']
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
    terminal_path = ['src/gan/data/', 'gan/data/', '../gan/data/', 'data/', '../data']
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

def load(filename):
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
    terminal_path = ['src/gan/data/', 'gan/data/', '../gan/data/', 'data/', '../data']
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
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]
    
def resize_dataset(dataset):
    """
        Resizes dataset of MNIST images to (32, 32).

        Parameters:
        -dataset: a numpy array of size [?, 1, 28, 28].
    """        
    args = [dataset[i:i+1000] for i in range(0, len(dataset), 1000)]
    
    def f(chunk):
        return transform.resize(chunk, (chunk.shape[0], 1, 32, 32))

    with cf.ThreadPoolExecutor() as executor:
        res = executor.map(f, args)
    
    res = np.array([*res])
    res = res.reshape(-1, 1, 32, 32)
    return res

def dataloader(X, y, BATCH_SIZE):
    """
        Returns a data generator.

        Parameters:
        - X: dataset examples.
        - y: ground truth labels.
    """
    n = len(X)
    for t in range(0, n, BATCH_SIZE):
        yield X[t:t+BATCH_SIZE, ...], y[t:t+BATCH_SIZE, ...]

def save_params_to_file(model):
    """
        Saves model parameters to a file.

        Parameters:
        -model: a CNN architecture.
    """
    # Make save_weights/ accessible from every folders.
    terminal_path = ["src/fast/save_weights/", "fast/save_weights/", '../fast/save_weights/', "save_weights/", "../save_weights/"]
    dirPath = None
    for path in terminal_path:
        if os.path.isdir(path):
            dirPath = path
    if dirPath == None:
        raise FileNotFoundError("save_params_to_file(): Impossible to find save_weights/ from current folder. You need to manually add the path to it in the \'terminal_path\' list and the run the function again.")

    weights = model.get_params()
    if dirPath == '../fast/save_weights/': # We run the code from demo notebook.
        with open(dirPath + "demo_weights.pkl","wb") as f:
            pickle.dump(weights, f)
    else:
        with open(dirPath + "final_weights.pkl","wb") as f:
            pickle.dump(weights, f)

def load_params_from_file(model, isNotebook=False):
    """
        Loads model parameters from a file.

        Parameters:
        -model: a CNN architecture.
    """
    if isNotebook: # We run from demo-notebooks/
        pickle_in = open("../fast/save_weights/demo_weights.pkl", 'rb')
        params = pickle.load(pickle_in)
        model.set_params(params)
    else:
        # Make final_weights.pkl file accessible from every folders.
        terminal_path = ["src/fast/save_weights/final_weights.pkl", "fast/save_weights/final_weights.pkl",
        "save_weights/final_weights.pkl", "../save_weights/final_weights.pkl"]

        filePath = None
        for path in terminal_path:
            if os.path.isfile(path):
                filePath = path
        if filePath == None:
            raise FileNotFoundError('load_params_from_file(): Cannot find final_weights.pkl from your current folder. You need to manually add it to terminal_path list and the run the function again.')

        pickle_in = open(filePath, 'rb')
        params = pickle.load(pickle_in)
        model.set_params(params)
    return model
            
def prettyPrint3D(M):
    """
        Displays a 3D matrix in a pretty way.

        Parameters:
        -M: Matrix of shape (m, n_H, n_W, n_C) with m, the number 3D matrices.
    """
    m, n_C, n_H, n_W = M.shape

    for i in range(m):
        
        for c in range(n_C):
            print('Image {}, channel {}'.format(i + 1, c + 1), end='\n\n')  

            for h in range(n_H):
                print("/", end="")

                for j in range(n_W):

                    print(M[i, c, h, j], end = ",")

                print("/", end='\n\n')
        
        print('-------------------', end='\n\n')

def plot_example(X, y, y_pred=None):
    """
        Plots 9 examples and their associate labels.
        
        Parameters:
        -X: Training examples.
        -y: true labels.
        -y_pred: predicted labels.
    """
    # Create figure with 3 x 3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
     
    X, y = X[:9, 0, ...], y[:9] 
    
    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(X[i])

        # Show true and predicted classes.
        if y_pred is None:
            xlabel = "True: {0}".format(y[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(y[i], y_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots in a single Notebook cell.
    plt.show()

def plot_example_errors(X, y, y_pred):
    """
        Plots 9 example errors and their associate true/predicted labels.
        
        Parameters:
        -X: Training examples.
        -y: true labels.
        -y_pred: predicted labels.
    
    """
    incorrect = (y != y_pred)
 
    X = X[incorrect]
    y = y[incorrect]
    y_pred = y_pred[incorrect]

    # Plot the first 9 images.
    plot_example(X, y, y_pred)

def show_result(EPOCHS, path, show=False, save=False):
    z = np.random.randn((5*5, 100))
    test_images = G(z)
    
    size_figure_grid = 5

    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(5*5):
        i = k // 5
        j = k % 5
        ax[i, j].cla()
        ax[i, j].imshow(test_images[k, :].cpu().data.view(28, 28).numpy(), cmap='gray')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close())

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
