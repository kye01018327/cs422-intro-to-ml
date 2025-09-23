# %%
import numpy as np
from scipy.spatial import distance

# %%
def calculate_sorted_distances(X, unknown_point):
    distances = [distance.euclidean(x, unknown_point) for x in X]
    sorted_idx = np.argsort(distances)
    if np.isclose(distances[0], 0): # Exclude same point
        sorted_idx = sorted_idx[1:]
    return sorted_idx, distances

def calculate_nearest_neighbors(X, Y, unknown_point, K):
    sorted_idx, distances = calculate_sorted_distances(X, unknown_point)
    nearest_neighbors = [Y[idx] for idx in sorted_idx[:K]]
    return nearest_neighbors

def KNN_predict(X, Y, unknown_point, K):
    nearest_neighbors = calculate_nearest_neighbors(X, Y, unknown_point, K)
    classes, counts = np.unique(nearest_neighbors, return_counts=True)
    max_count = np.max(counts)
    classes_with_max_count = classes[counts == max_count]
    if len(classes_with_max_count) == 1:
        return classes_with_max_count[0]
    else: # Select closest neighbor in tiebreaker event
        return nearest_neighbors[0]


def KNN_test(X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray, Y_test: np.ndarray, K: int):
    Y_predictions = []
    # Compile predictions for each point
    for unknown_point in X_test:
        prediction = KNN_predict(X_train, Y_train, unknown_point, K)
        Y_predictions.append(prediction)

    # Compare predictions (Y_predictions) to expected labels (Y_test)
    intersection = Y_predictions == Y_test
    accuracy = sum(intersection) / len(Y_test)
    return accuracy

