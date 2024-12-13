# Convolutional Neural Network (CNN) Implementation from Scratch

import numpy as np

# Function to perform convolution operation
def convolve2d(input_matrix, kernel):
    kernel_height, kernel_width = kernel.shape
    input_height, input_width = input_matrix.shape
    
    output_height = input_height - kernel_height + 1
    output_width = input_width - kernel_width + 1
    
    output = np.zeros((output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):
            region = input_matrix[i:i + kernel_height, j:j + kernel_width]
            output[i, j] = np.sum(region * kernel)

    return output

# Function to perform ReLU activation
def relu(matrix):
    return np.maximum(0, matrix)

# Function to perform max pooling
def max_pooling(matrix, pool_size, stride):
    input_height, input_width = matrix.shape
    
    output_height = (input_height - pool_size) // stride + 1
    output_width = (input_width - pool_size) // stride + 1
    
    output = np.zeros((output_height, output_width))

    for i in range(0, output_height):
        for j in range(0, output_width):
            region = matrix[
                i * stride:i * stride + pool_size,
                j * stride:j * stride + pool_size
            ]
            output[i, j] = np.max(region)

    return output

# Example CNN forward pass
if __name__ == "__main__":
    # Example input (5x5 image)
    input_image = np.array([
        [1, 2, 3, 0, 1],
        [0, 1, 2, 3, 0],
        [1, 0, 1, 2, 3],
        [3, 1, 0, 1, 2],
        [2, 3, 1, 0, 1]
    ])

    # Example kernel (3x3)
    kernel = np.array([
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1]
    ])

    # Convolution operation
    convolved_output = convolve2d(input_image, kernel)
    print("Convolved Output:")
    print(convolved_output)

    # ReLU activation
    relu_output = relu(convolved_output)
    print("ReLU Output:")
    print(relu_output)

    # Max pooling (2x2 pooling with stride 2)
    pooled_output = max_pooling(relu_output, pool_size=2, stride=2)
    print("Max Pooled Output:")
    print(pooled_output)
