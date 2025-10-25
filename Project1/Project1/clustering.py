# %%
import numpy as np
from scipy.spatial import distance
import random

# %%
def randomize(X, K):
    # Fisher-Yates shuffle
    X_indices = np.arange(len(X))
    for i in range(len(X_indices) - 1, 0, -1):
        j = random.randint(0, i) 
        X_indices[i], X_indices[j] = X_indices[j], X_indices[i]
    return X_indices[:K]

def recalculate_centroids(X, mu):
    # Create list of empty clusters
    clusters = [[] for _ in mu]

    # Iterate through each point in the training set (X)
    for point in X:
        # Get distance between point and each centroid
        distances = []
        for cluster in mu:
            distances.append(distance.euclidean(point, cluster))
        # Select closest cluster (if same distance, select first occurrence or lowest index)
        closest_cluster_idx = np.argmin(distances) # Potential flaw for same distances
        # Append point to the closest cluster
        clusters[closest_cluster_idx].append(point)

    # Create a new centroid
    new_mu = []
    # Iterate through clusters
    for idx, cluster in enumerate(clusters):
        # If cluster has points, calculate new centroid for that cluster
        if cluster:
            new_mu.append(np.mean(clusters[idx], axis=0))
        # Else keep the old centroid
        else:
            new_mu.append(mu[idx])

    new_mu = np.array(new_mu)

    return new_mu

def K_Means(X: np.ndarray, K, mu: np.ndarray):
    mu = list(mu)
    # If centroids are uninitialized, randomize starting points for K amount of clusters
    if not mu:
        random_starting_points = randomize(X, K)
        mu = X[random_starting_points]

    # Recalculate centroids until convergence
    new_mu = mu
    converged = False
    while not converged:
        mu = new_mu
        new_mu = recalculate_centroids(X, mu)
        converged = np.allclose(new_mu, mu)

    return mu
    
