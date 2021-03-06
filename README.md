# UFLDL-Tutorial
---

This repository contains the MATLAB implementation of the exersises from the [Unsupervised Feature Learning and Deep Learning](http://ufldl.stanford.edu/wiki/index.php/UFLDL_Tutorial) (UFLDL) Tutorial.

#### Notes about the input data

You can find the appropriate data for each exercise at:

  1. For the sparseae_exercise, you do not need to do anythings as the data are already included (it is the `IMAGES.mat` file).
  2. For the sparseae_exercise_vector, you do not need to do anything as the data are already included (it is the `IMAGES.mat` file).
  3. For the pca_2d, you do not need to do anything as the data are already included (it is the `pcaData.txt` file).
  4. For the pca_exercise, you do not need to do anything as the data are already included (it is the `IMAGES_RAW.mat` file).
  5. For the softmax_exercise, you need to download the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). Specifically, you need to download:
    * The `train-images-idx3-ubyte.gz`
    * The `train-labels-idx1-ubyte.gz`
    * The `t10k-images-idx3-ubyte.gz`
    * The `t10k-labels-idx1-ubyte.gz`
    More information can be found on the tutorial's [webpage](http://ufldl.stanford.edu/wiki/index.php/Exercise:Softmax_Regression).
  6. For the stl_exercise, you need to download the same dataset as in the softmax_exercise (see 5).

#### A note about the license

The initial code for the excercises is available at the [UFLDL Tutorial](http://ufldl.stanford.edu/wiki/index.php/UFLDL_Tutorial), and my code is provided here under the MIT licence. **But**, as noted in the [Sparse Autoencoder exercise](http://ufldl.stanford.edu/wiki/index.php/Exercise:Sparse_Autoencoder), xthe minFunc subdirectory is a 3rd party software that implements the L-BFGS optimization algorithm and is licensed under a Creative Commons, Attribute, Non-Commercial licence. If you need to use this software for commercial purposes, you can download and use a different functions (fminlbfgs) that can serve the same purpose, but runs ~3x slower. You can find more about this function [here](http://ufldl.stanford.edu/wiki/index.php/Fminlbfgs_Details).
