# Deep Learning — Core Concepts

## Chapter 1: Neural Network Fundamentals

A neural network is composed of layers of interconnected neurons. Each neuron receives input from the previous layer, applies a weighted sum followed by a non-linear activation function, and passes the result to the next layer. The simplest form is the feedforward neural network, where information flows in one direction without cycles.

### Activation Functions

The ReLU (Rectified Linear Unit) activation function is defined as f(x) = max(0, x). It has become the default activation function for many neural network architectures because it helps mitigate the vanishing gradient problem. Other activation functions include the sigmoid function, which squashes values to the range (0, 1), and the tanh function, which squashes values to the range (-1, 1).

### Loss Functions

The choice of loss function depends on the task. For regression tasks, mean squared error is commonly used. For classification tasks, cross-entropy loss is the standard choice. The loss function measures how far the network's predictions are from the ground truth.

## Chapter 2: Training Neural Networks

### Backpropagation

Backpropagation is the algorithm for computing gradients of the loss function with respect to the network parameters. It works by applying the chain rule from calculus to propagate error signals backward through the network. Once the gradients are computed, an optimization algorithm such as stochastic gradient descent (SGD) uses them to update the parameters.

### Stochastic Gradient Descent

SGD updates parameters in the direction opposite to the gradient of the loss function. The learning rate controls the step size of each update. Mini-batch SGD is a variant that computes gradients on small random subsets of the training data rather than the full dataset.

### Regularization

Regularization techniques prevent overfitting. L1 regularization adds the absolute value of weights to the loss function, encouraging sparse weight matrices. L2 regularization (also known as weight decay) adds the squared magnitude of weights to the loss. Dropout is a regularization technique where randomly selected neurons are omitted during training, forcing the network to learn redundant representations.

## Chapter 3: Convolutional Neural Networks

Convolutional neural networks are designed to process grid-like data such as images. A convolution operation applies a filter (also called a kernel) across the input, producing a feature map. Key properties of convolutions include local connectivity, parameter sharing, and translational equivariance.

### Pooling Layers

Pooling layers reduce the spatial dimensions of feature maps. Max pooling selects the maximum value in each pooling window, while average pooling computes the mean. Pooling provides a form of translational invariance and reduces the computational cost of subsequent layers.

### Common Architectures

LeNet-5 was one of the earliest CNN architectures for handwritten digit recognition. AlexNet demonstrated the effectiveness of deep CNNs on large-scale image classification. ResNet introduced skip connections (residual connections) that allow gradients to flow directly through the network, enabling training of very deep architectures.

## Chapter 4: Recurrent Neural Networks

Recurrent neural networks are designed for sequential data such as text or time series. An RNN maintains a hidden state that is updated at each time step based on the current input and the previous hidden state. This allows the network to capture temporal dependencies.

### The Vanishing Gradient Problem

Standard RNNs struggle to capture long-range dependencies because gradients tend to either vanish or explode as they are backpropagated through many time steps. This is known as the vanishing gradient problem. The hidden state of an RNN at early time steps has diminishing influence on the gradients at later time steps.

### LSTM and GRU

Long Short-Term Memory networks introduce a gating mechanism with input, forget, and output gates that control the flow of information. The cell state in an LSTM acts as a memory that can be preserved across many time steps. Gated Recurrent Units are a simplified variant with only two gates: reset and update.

## Chapter 5: Transformers and Attention

### The Attention Mechanism

The attention mechanism allows a model to focus on relevant parts of the input when producing each element of the output. In the context of sequence-to-sequence models, attention computes a weighted sum of encoder hidden states, where the weights are determined by a compatibility function between the decoder state and each encoder state.

### Self-Attention

Self-attention, also known as intra-attention, computes attention within a single sequence. Each position in the sequence attends to all other positions. The Transformer architecture is built entirely on self-attention, without recurrence or convolution.

### Multi-Head Attention

Multi-head attention runs multiple attention operations in parallel, each with different learned projections. The outputs are concatenated and linearly transformed. This allows the model to attend to information from different representation subspaces at different positions.

### Transformer Architecture

The Transformer consists of an encoder and a decoder, each composed of a stack of identical layers. Each layer has two sub-layers: a multi-head self-attention mechanism and a position-wise feedforward network. Residual connections and layer normalization are applied around each sub-layer. Positional encodings are added to the input embeddings to provide information about the position of each token in the sequence.
