<img src="./img/logo.png" hspace="30%" width="40%">

## Introduction

- **GANumpy** is a Generative Adversarial Network written in pure Numpy (educational purpose only).
- It uses [Yaae][yaae], my custom **automatic differentiation engine** also written in pure Numpy.
- The GAN was trained on MNIST dataset. To speed up the training time, only a subset of digit samples (1,2,3) were used. Here are the results:

<img src="./src/MNIST_GAN_results/generation_animation.gif" hspace="25%" width="50%">

## Installation

- Create a virtual environment in the root folder using [virtualenv][virtualenv] and activate it.

```bash
# On Linux terminal, using virtualenv.
virtualenv my_ganumpy_env
# Activate it.
source my_ganumpy_env/bin/activate
```

- Install **requirements.txt**.

```bash
pip install -r requirements.txt
# Tidy up the root folder.
python3 setup.py clean
```

<!---
Variables with links.
-->
[yaae]: https://github.com/3outeille/Yaae
[virtualenv]: https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/
