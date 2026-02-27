# Building a Neural Network in C to guess the handwitten number

Im using 784 input pixels, 100 neuons in the hidden layer, and 10 in the output for the ints 0-9. This is in my case the functions aren't hardcoded.

Im initializing their weights and bias' with random values, weights = He scale, bias = [-0.001, 0.001]

Using ReLU on the hidden layer, and softmax on the output layer.

Using Categorial Corss-Entropy Loss for the 10 classes i have to calculate the loss.

For optimization im using Stochastic Gradient Descent for one sample at a time, with a learning rate of 0.02 - 5% each epoch.

Using chace friendly data structes such as a collapsed list of weights, and implemented SIMD and parallelization to improve the speed of the training.

Im getting ~98.3% accuracy rate on testing.

## Completed improvements

- [x] Separate training and Inference
- [x] Take binary files as input instead of CSV for faster reading
- [x] He initialization for ReLU
- [x] Skipping softmax and loss calc at Inference and using argmax for optimization
- [x] Implement Batch trainging  (Was worse)
- [x] Learning rate scheduling (changing the learning rate over time)
- [x] Move malloc/free out of the loop so it isnt called 60.000+ times
- [x] Allow input from my own drawings
- [x] Made a nice GUI to interact with it
- [X] Implemented SIMD which gave a 25-30% speedup

## For the future i want to

- [ ] Improve chace locality on backprop using a transposed weigth matrix (collapsed ofc)
- [ ] Expand to letters aswell using the EMNIST dataset
- [ ] More layers generalized
- [ ] Dropout regularization (layer-wise)
- [ ] SGD with momentum (0.9) for better convergence

Dataset from MNIST
