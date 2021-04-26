This code contains a C++ implementation of full-forward multi-layer perceptron using Eigen Library.

Implementation details:

The working of a multi-layer perceptron is defined in the file "multi_layer_perceptron.h". The Layer class incorporates a layer of MLP. It has 4 subclasses: Dense Layer, RELU, Sigmoid and Tanh representing different type of layers in the neural network. The code divides the data into batches and then train neural network onto it using backpropagation. The linear algebra calculations are accomplished using Eigen library.

The "main.cpp" provides interface to the user to generate a neural network model by giving training and testing data and defining the network. The default data used here is MNIST (Digit recognition) data minimized to just 900 training and 100 testing images.


How to use:

Call the makefile using "make" command. It will generate the executable and run it.

