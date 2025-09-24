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

def recalculate_clusters(X, mu):
    clusters = [[] for _ in mu]

    for point in X:
        # Get distance between point and each cluster
        distances = []
        for cluster in mu:
            distances.append(distance.euclidean(point, cluster))
        # Select closest cluster

        closest_cluster_idx = np.argmin(distances) # Potential flaw for same distances
        clusters[closest_cluster_idx].append(point)

    for idx, cluster in enumerate(clusters):
        cluster = np.array(cluster)

    new_mu = []
    for idx, cluster in enumerate(clusters):
        if cluster:
            new_mu.append(np.mean(clusters[idx], axis=0))
        else:
            new_mu.append(mu[idx])

    new_mu = np.array(new_mu)

    return new_mu

def K_Means(X: np.ndarray, K, mu: np.ndarray):
    mu = list(mu)
    # Randomize starting points for K amount of Clusters
    if not mu:
        X_size = len(X)
        random_starting_points = randomize(X, K)
        mu = X[random_starting_points]

    new_mu = mu
    converged = False
    while not converged:
        mu = new_mu
        new_mu = recalculate_clusters(X, mu)
        converged = np.allclose(new_mu, mu)

    return mu
    
