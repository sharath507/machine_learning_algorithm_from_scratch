# Lasso Regression Implementation from Scratch
# Lasso Regression adds L1 regularization to the cost function to encourage sparsity in coefficients.

# Function to calculate the mean of a list
def mean(values):
    return sum(values) / len(values)

# Function to calculate coefficients with L1 regularization (Lasso)
def lasso_coefficients(X, Y, alpha, n_iterations, learning_rate):
    n_samples = len(Y)
    n_features = len(X[0])

    # Initialize coefficients and intercept
    weights = [0.0] * n_features
    bias = 0.0

    for _ in range(n_iterations):
        # Update each weight and bias
        for i in range(n_samples):
            prediction = bias + sum(weights[j] * X[i][j] for j in range(n_features))
            error = prediction - Y[i]

            # Update bias
            bias -= learning_rate * error

            # Update weights with L1 regularization
            for j in range(n_features):
                gradient = error * X[i][j]

                if weights[j] > 0:
                    weights[j] -= learning_rate * (gradient + alpha)
                elif weights[j] < 0:
                    weights[j] -= learning_rate * (gradient - alpha)
                else:
                    weights[j] -= learning_rate * gradient

    return bias, weights

# Function to predict Y values based on the regression line
def predict(X, bias, weights):
    return [bias + sum(weights[j] * x[j] for j in range(len(weights))) for x in X]

# Sample Dataset
X = [[1], [2], [3], [4], [5]]  # Feature matrix
Y = [1.2, 2.3, 2.9, 3.8, 5.1]  # Target values

# Hyperparameters
alpha = 0.1  # Regularization strength
n_iterations = 1000
learning_rate = 0.01

# Train the Lasso Regression model
bias, weights = lasso_coefficients(X, Y, alpha, n_iterations, learning_rate)

# Predictions
predictions = predict(X, bias, weights)

# Output the results
print(f"Trained Weights: {weights}")
print(f"Trained Bias: {bias}")
print(f"Predictions: {predictions}")
