# Linear Discriminant Analysis (LDA) Implementation from Scratch
# LDA projects data onto a lower-dimensional space to maximize class separability.

import numpy as np

# Step 1: Compute the mean vectors for each class
def compute_class_means(X, y):
    class_labels = np.unique(y)
    means = {}
    for label in class_labels:
        means[label] = np.mean(X[y == label], axis=0)
    return means

# Step 2: Compute the within-class scatter matrix
def within_class_scatter_matrix(X, y, class_means):
    n_features = X.shape[1]
    S_w = np.zeros((n_features, n_features))
    class_labels = np.unique(y)

    for label in class_labels:
        class_scatter = np.zeros((n_features, n_features))
        for row in X[y == label]:
            row_diff = (row - class_means[label]).reshape(n_features, 1)
            class_scatter += np.dot(row_diff, row_diff.T)
        S_w += class_scatter

    return S_w

# Step 3: Compute the between-class scatter matrix
def between_class_scatter_matrix(X, y, class_means):
    n_features = X.shape[1]
    overall_mean = np.mean(X, axis=0)
    S_b = np.zeros((n_features, n_features))
    class_labels = np.unique(y)

    for label in class_labels:
        n_samples = X[y == label].shape[0]
        mean_diff = (class_means[label] - overall_mean).reshape(n_features, 1)
        S_b += n_samples * np.dot(mean_diff, mean_diff.T)

    return S_b

# Step 4: Compute eigenvalues and eigenvectors to find the linear discriminants
def compute_lda(X, y, n_components):
    class_means = compute_class_means(X, y)
    S_w = within_class_scatter_matrix(X, y, class_means)
    S_b = between_class_scatter_matrix(X, y, class_means)

    # Solve the generalized eigenvalue problem for S_w^(-1) * S_b
    eigvals, eigvecs = np.linalg.eig(np.linalg.inv(S_w).dot(S_b))

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigvals)[::-1]
    eigvals = eigvals[sorted_indices]
    eigvecs = eigvecs[:, sorted_indices]

    # Select the top `n_components` eigenvectors
    linear_discriminants = eigvecs[:, :n_components]

    return linear_discriminants

# Step 5: Project the data onto the linear discriminants
def project_data(X, linear_discriminants):
    return np.dot(X, linear_discriminants)

# Sample Dataset
X = np.array([
    [4.0, 2.0],
    [2.0, 4.0],
    [2.0, 3.0],
    [3.0, 6.0],
    [4.0, 4.0],
    [9.0, 10.0],
    [6.0, 8.0],
    [9.0, 5.0],
    [8.0, 7.0],
    [10.0, 8.0]
])
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])  # Class labels

# Number of components to keep
n_components = 1

# Perform LDA
linear_discriminants = compute_lda(X, y, n_components)
projected_data = project_data(X, linear_discriminants)

# Output the results
print("Linear Discriminants:")
print(linear_discriminants)
print("Projected Data:")
print(projected_data)
