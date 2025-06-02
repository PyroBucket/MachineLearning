import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import time

# Activation functions and their derivatives
activation_functions = {
    "sigmoid": (lambda x: 1 / (1 + np.exp(-x)), lambda x: x * (1 - x)),
    "relu": (lambda x: np.maximum(0, x), lambda x: np.where(x > 0, 1, 0)),
    "tanh": (lambda x: np.tanh(x), lambda x: 1 - np.tanh(x)**2),
    "softmax": (lambda x: np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True), None)
}

# Loss functions and their derivatives
loss_functions = {
    "cross_entropy": (lambda y, y_pred: -np.mean(np.sum(y * np.log(y_pred + 1e-8), axis=1)),
                       lambda y, y_pred: y_pred - y),
    "mse": (lambda y, y_pred: np.mean(np.square(y - y_pred)),
             lambda y, y_pred: 2 * (y_pred - y) / y.shape[0])
}

# Multilayer Perceptron Implementation
class MLP:
    def __init__(self, input_size, layer_sizes, activations, loss, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.loss_fn, self.loss_derivative = loss_functions[loss]
        self.activations = [activation_functions[act] for act in activations]

        # Initialize weights and biases for multiple layers
        self.weights = []
        self.biases = []

        # Input to first layer
        self.weights.append(np.random.randn(input_size, layer_sizes[0]) * 0.01)
        self.biases.append(np.zeros((1, layer_sizes[0])))

        # Hidden layers
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.01)
            self.biases.append(np.zeros((1, layer_sizes[i + 1])))

    def forward(self, X):
        # Forward pass through all layers
        self.inputs = []
        self.outputs = []

        input_data = X
        for i in range(len(self.weights)):
            self.inputs.append(input_data)
            z = np.dot(input_data, self.weights[i]) + self.biases[i]
            activation, _ = self.activations[i]
            a = activation(z)
            self.outputs.append(a)
            input_data = a

        return self.outputs[-1]

    def backward(self, X, y):
        # Backward pass
        m = y.shape[0]

        # Output layer error
        output_error = self.loss_derivative(y, self.outputs[-1])
        _, output_derivative = self.activations[-1]
        delta = output_error * (output_derivative(self.outputs[-1]) if output_derivative else 1)

        # Gradients for last layer
        self.weights[-1] -= self.learning_rate * np.dot(self.outputs[-2].T, delta) / m
        self.biases[-1] -= self.learning_rate * np.sum(delta, axis=0, keepdims=True) / m

        # Backpropagate through hidden layers
        for i in range(len(self.weights) - 2, -1, -1):
            output_derivative = self.activations[i][1]
            hidden_error = np.dot(delta, self.weights[i + 1].T)
            delta = hidden_error * (output_derivative(self.outputs[i]) if output_derivative else 1)

            input_layer = self.inputs[i]
            self.weights[i] -= self.learning_rate * np.dot(input_layer.T, delta) / m
            self.biases[i] -= self.learning_rate * np.sum(delta, axis=0, keepdims=True) / m

    def train(self, X, y, epochs):
        start_time = time.time()
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y)
            if epoch % 100 == 0:
                loss = self.loss_fn(y, self.outputs[-1])
                print(f"Epoch {epoch}, Loss: {loss}")
        end_time = time.time()
        print(f"Training Time: {end_time - start_time:.2f} seconds")

    def predict(self, X):
        output = self.forward(X)
        if self.activations[-1][0] == activation_functions["softmax"][0]:
            return np.argmax(output, axis=1)
        return (output > 0.5).astype(int)

# Load the MNIST dataset
data = load_digits()
X = data.data  # Features (flattened images)
y = data.target  # Labels

# One-hot encode the labels
y_one_hot = np.zeros((y.size, y.max() + 1))
y_one_hot[np.arange(y.size), y] = 1

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y_one_hot, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Normalize the data (scaling to mean=0 and variance=1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Define and train the MLP model
input_size = X_train.shape[1]
layer_sizes = [64,128, 10]  # Two hidden layers and one output layer
activations = ["relu", "relu", "softmax"]
loss = "cross_entropy"
learning_rate = 0.1

mlp = MLP(input_size, layer_sizes, activations, loss, learning_rate)
mlp.train(X_train, y_train, epochs=1000)

# Evaluate the model on the validation set
y_val_pred = mlp.predict(X_val)
y_val_true = np.argmax(y_val, axis=1)
print("Validation Accuracy:", accuracy_score(y_val_true, y_val_pred))

# Evaluate the model on the test set
y_test_pred = mlp.predict(X_test)
y_test_true = np.argmax(y_test, axis=1)
print("Test Accuracy:", accuracy_score(y_test_true, y_test_pred))

# Print a detailed classification report
print("Classification Report:")
print(classification_report(y_test_true, y_test_pred))
