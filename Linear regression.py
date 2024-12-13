# Linear Regression Implementation from Scratch

# Function to calculate the mean of a list
def mean(values):
    return sum(values) / len(values)

# Function to calculate the covariance between X and Y
def covariance(X, Y, X_mean, Y_mean):
    cov = 0.0
    for i in range(len(X)):
        cov += (X[i] - X_mean) * (Y[i] - Y_mean)
    return cov

# Function to calculate the variance of X
def variance(X, X_mean):
    var = 0.0
    for x in X:
        var += (x - X_mean) ** 2
    return var

# Function to calculate coefficients (slope and intercept)
def coefficients(X, Y):
    X_mean = mean(X)
    Y_mean = mean(Y)
    b1 = covariance(X, Y, X_mean, Y_mean) / variance(X, X_mean)  # Slope
    b0 = Y_mean - b1 * X_mean  # Intercept
    return b0, b1

# Function to predict Y values based on the regression line
def predict(X, b0, b1):
    return [b0 + b1 * x for x in X]

# Sample Dataset
X = [1, 2, 3, 4, 5]
Y = [2, 4, 5, 4, 5]

# Calculate coefficients
b0, b1 = coefficients(X, Y)

# Predict values
predictions = predict(X, b0, b1)

# Output the results
print(f"Slope (b1): {b1}")
print(f"Intercept (b0): {b0}")
print(f"Predicted values: {predictions}")
