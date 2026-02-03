# Building a Neural Network in C to guess the handwitten number.

Im using 784 input pixels, 100 neuons in the hidden layer, and 10 in the output for the ints 0-9.

Im initializing their weights and bias' with random values, weights = [-0.01, 0.01], bias = [-0.001, 0.001]

Using ReLU on the hidden layer, and softmax on the output layer.

Using Categorial Corss-Entropy Loss for the 10 classes i have to calculate the loss.

For optimization im using Stochastic Gradient Descent for one sample at a time, with a fixed learning rate of 0.01 rn.



## For the future i want to:
- [x] Separate training and Inference 
- [x] Take binary files as input instead of CSV for faster reading
- [ ] Skipping softmax and loss calc at Inference and using argmax for optimization 
- [ ] Allow input from my own drawings
- [ ] Implement Batch trainging  (almost done)
- [ ] Learning rate scheduling (changing the learning rate over time)
- [ ] More hidden layers for deeper learning
- [ ] Expand to letters aswell if i can find such a dataset


Dataset from MNIST
