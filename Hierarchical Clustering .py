# Hierarchical Clustering Implementation from Scratch

import math

# Function to calculate Euclidean distance
def euclidean_distance(point1, point2):
    return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)))

# Function to calculate distance between two clusters
# Using the single-linkage method (minimum distance between clusters)
def single_linkage_distance(cluster1, cluster2):
    return min(euclidean_distance(p1, p2) for p1 in cluster1 for p2 in cluster2)

# Function to perform Hierarchical Clustering
def hierarchical_clustering(data, num_clusters):
    # Initialize each point as its own cluster
    clusters = [[point] for point in data]

    while len(clusters) > num_clusters:
        # Find the two closest clusters
        min_distance = float('inf')
        closest_pair = (None, None)

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                distance = single_linkage_distance(clusters[i], clusters[j])
                if distance < min_distance:
                    min_distance = distance
                    closest_pair = (i, j)

        # Merge the two closest clusters
        cluster1, cluster2 = closest_pair
        clusters[cluster1].extend(clusters[cluster2])
        del clusters[cluster2]

    return clusters

# Example usage
data_points = [
    [1, 2], [2, 3], [3, 4], [8, 8], [9, 9], [10, 10], [25, 25]
]
num_clusters = 2

clusters = hierarchical_clustering(data_points, num_clusters)
print("Clusters:")
for idx, cluster in enumerate(clusters):
    print(f"Cluster {idx + 1}: {cluster}")
