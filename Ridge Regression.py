# Ridge Regression Implementation from Scratch
# Ridge Regression adds L2 regularization to the cost function to prevent overfitting.

# Function to calculate the mean of a list
def mean(values):
    return sum(values) / len(values)

# Function to calculate coefficients with L2 regularization
def ridge_coefficients(X, Y, alpha):
    X_mean = mean(X)
    Y_mean = mean(Y)

    # Calculate covariance and variance
    cov = sum((X[i] - X_mean) * (Y[i] - Y_mean) for i in range(len(X)))
    var = sum((X[i] - X_mean) ** 2 for i in range(len(X)))

    # Ridge regression adjustment to slope (b1)
    b1 = cov / (var + alpha)  # Adding regularization term alpha to the denominator

    # Intercept (b0)
    b0 = Y_mean - b1 * X_mean

    return b0, b1

# Function to predict Y values based on the regression line
def predict(X, b0, b1):
    return [b0 + b1 * x for x in X]

# Sample Dataset
X = [1, 2, 3, 4, 5]
Y = [2, 4, 5, 4, 5]
alpha = 1.0  # Regularization strength

# Calculate coefficients
b0, b1 = ridge_coefficients(X, Y, alpha)

# Predict values
predictions = predict(X, b0, b1)

# Output the results
print(f"Ridge Regression Coefficients:")
print(f"Slope (b1): {b1}")
print(f"Intercept (b0): {b0}")
print(f"Predicted values: {predictions}")
