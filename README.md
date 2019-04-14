# MNISTCNTK
A MNIST test program written by CNTK with CNN

## Current best score
Now the lastest model rans a best score, it list as follow.
### Lastest Model
* Update the convolution activation function from ReLU to LakeyReLU.

Error rate in offical test dataset down to 0.002%!(model file is 'cnn-model-errorrate0.001.mld', with the program bug that my error rate showed 0.0019(0.019%)).

Testing result:
* Testing Dataset Size: 10000
* Finally Misclassification: 2
* Accuracy Rate: 99.98%

The training rans on my workstation(with Intel(R) Xeon(R) E5-2689v4(10C20T@3.4GHz) x1 CPU, no GPU) needed about 20 minutes.

Training parameters:
* Algorithm: Adam Gradient Descent
* Minibatch Size: 35
* Total Data Size: 60000
* Round: 15

### Current CNN Structure
My network simulated on Mathematica 11.2.

The simple structure is list as follow.

* Input (Normalizated grayscale image 28 x 28)
* Convolution layer I: Kernel: 3 x 3 with Stride 1, Channel: 30
* LakeyReLU
* Pooling layer I: Kernel: 3 x 3 with Srride 2
* Convolution layer II: Kernel: 3 x 3 with Stride 1, Channel: 10
* LakeyReLU
* Pooling layer II: Kernel: 3 x 3 with Srride 1
* Flatten layer
* Fully connected layer I: 240 Neurons
* Sigmoid
* Fully connected layer II: 100 Neurons
* Sigmoid
* Fully connected layer III: 10 Neurons
* Softmax classfication layer

## Project information
### Framework
.NET Core 3.0 Preview
Microsoft CNTK 2.6(NuGet Package)
C# 7.0
