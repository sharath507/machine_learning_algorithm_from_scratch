# K-Means Clustering Implementation from Scratch
import random

# Function to calculate Euclidean distance
def euclidean_distance(point1, point2):
    return sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)) ** 0.5

# Function to initialize centroids randomly
def initialize_centroids(dataset, k):
    return random.sample(dataset, k)

# Function to assign clusters based on centroids
def assign_clusters(dataset, centroids):
    clusters = [[] for _ in range(len(centroids))]
    for point in dataset:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        cluster_index = distances.index(min(distances))
        clusters[cluster_index].append(point)
    return clusters

# Function to compute new centroids
def compute_centroids(clusters):
    centroids = []
    for cluster in clusters:
        if cluster:
            centroid = [sum(dim) / len(cluster) for dim in zip(*cluster)]
            centroids.append(centroid)
        else:
            centroids.append([])
    return centroids

# Function to check for convergence
def is_converged(old_centroids, new_centroids):
    return all(euclidean_distance(old, new) < 1e-6 for old, new in zip(old_centroids, new_centroids))

# K-Means Clustering Algorithm
def kmeans(dataset, k, max_iterations=100):
    centroids = initialize_centroids(dataset, k)
    for _ in range(max_iterations):
        clusters = assign_clusters(dataset, centroids)
        new_centroids = compute_centroids(clusters)
        if is_converged(centroids, new_centroids):
            break
        centroids = new_centroids
    return clusters, centroids

# Sample Dataset
dataset = [
    [2.0, 3.0],
    [1.0, 4.0],
    [3.0, 4.0],
    [5.0, 8.0],
    [8.0, 8.0],
    [9.0, 10.0],
    [10.0, 5.0],
    [7.0, 5.0]
]

# Number of clusters
k = 2

# Run K-Means
clusters, centroids = kmeans(dataset, k)

# Display results
print("Final Centroids:", centroids)
for i, cluster in enumerate(clusters):
    print(f"Cluster {i + 1}: {cluster}")
