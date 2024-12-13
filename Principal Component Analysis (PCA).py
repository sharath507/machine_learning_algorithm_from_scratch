# Principal Component Analysis (PCA) Implementation from Scratch
# PCA reduces the dimensionality of the data while preserving as much variance as possible.

import numpy as np

# Function to compute the mean of each feature
def compute_mean(X):
    return np.mean(X, axis=0)

# Function to center the dataset
def center_data(X, mean):
    return X - mean

# Function to calculate the covariance matrix
def covariance_matrix(X):
    return np.dot(X.T, X) / (X.shape[0] - 1)

# Function to perform PCA
# `n_components` specifies how many principal components to keep
def pca(X, n_components):
    # Step 1: Compute the mean of each feature
    mean = compute_mean(X)

    # Step 2: Center the data
    centered_X = center_data(X, mean)

    # Step 3: Compute the covariance matrix
    cov_matrix = covariance_matrix(centered_X)

    # Step 4: Compute eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Step 5: Sort eigenvalues and eigenvectors in descending order of eigenvalues
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Step 6: Select the top `n_components` eigenvectors
    principal_components = eigenvectors[:, :n_components]

    # Step 7: Project the data onto the principal components
    reduced_data = np.dot(centered_X, principal_components)

    return reduced_data, principal_components

# Sample Dataset
X = np.array([
    [2.5, 2.4],
    [0.5, 0.7],
    [2.2, 2.9],
    [1.9, 2.2],
    [3.1, 3.0],
    [2.3, 2.7],
    [2.0, 1.6],
    [1.0, 1.1],
    [1.5, 1.6],
    [1.1, 0.9]
])

# Number of principal components to keep
n_components = 1

# Perform PCA
reduced_data, principal_components = pca(X, n_components)

# Output the results
print("Reduced Data:")
print(reduced_data)
print("Principal Components:")
print(principal_components)
