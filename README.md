Building a Neural Network in C to guess the handwitten number.
Dataset from MNIST

Im using 784 input pixels, 100 neuons in the hidden layer, and 10 in the output for the ints 0-9.

Im initializing their weights and bias' with random values, weights = [-0.01, 0.01], bias = [-0.001, 0.001]

Usin ReLU on the hidden layer, and softmax on the output layer.

Using Categorial Corss-Entropy Loss for the 10 classes i have to calculate the loss.

For optimization im using Stochastic Gradien Descent for one sample at a time, with a fixed learning rate of 0.01 rn.



For the future i want to:
\begin{itemize}
\item Train on data the network hansnt seen 
\item Implement Batch trainging
\item Learning rate scheduling (changing the learning rate over time)
\item More hidden layers for deeper learning
\end{itemize}
