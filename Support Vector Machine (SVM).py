# Support Vector Machine (SVM) Implementation from Scratch
# This implementation uses the primal form of SVM with gradient descent.

import numpy as np

# Function to initialize weights and bias
def initialize_weights(n_features):
    weights = np.zeros(n_features)
    bias = 0
    return weights, bias

# Function to compute the hinge loss
def hinge_loss(X, Y, weights, bias, C):
    n_samples = len(Y)
    loss = 0.5 * np.dot(weights, weights)  # Regularization term

    for i in range(n_samples):
        margin = Y[i] * (np.dot(weights, X[i]) + bias)
        loss += max(0, 1 - margin) * C

    return loss

# Function to perform gradient descent for SVM
def svm_train(X, Y, C, learning_rate, n_iterations):
    n_samples, n_features = X.shape
    weights, bias = initialize_weights(n_features)

    for _ in range(n_iterations):
        for i in range(n_samples):
            margin = Y[i] * (np.dot(weights, X[i]) + bias)

            if margin >= 1:
                # Update weights for correctly classified points
                weights -= learning_rate * weights
            else:
                # Update weights and bias for misclassified points
                weights -= learning_rate * (weights - C * Y[i] * X[i])
                bias += learning_rate * C * Y[i]

    return weights, bias

# Function to make predictions
def svm_predict(X, weights, bias):
    return np.sign(np.dot(X, weights) + bias)

# Sample Dataset (2D features for simplicity)
X = np.array([[1, 2], [2, 3], [3, 3], [2, 1], [3, 2]])  # Feature matrix
Y = np.array([1, 1, 1, -1, -1])  # Labels (1 or -1)

# Hyperparameters
C = 1.0  # Regularization strength
learning_rate = 0.01
n_iterations = 1000

# Train the SVM
weights, bias = svm_train(X, Y, C, learning_rate, n_iterations)

# Predictions
predictions = svm_predict(X, weights, bias)

# Output the results
print(f"Trained Weights: {weights}")
print(f"Trained Bias: {bias}")
print(f"Predictions: {predictions}")
