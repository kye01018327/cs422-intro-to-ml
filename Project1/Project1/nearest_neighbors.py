# %%
import numpy as np
from scipy.spatial import distance

# %%
def calculate_sorted_distances(X, unknown_point):
    # Calculate distances between unknown point and known points
    distances = [distance.euclidean(x, unknown_point) for x in X]
    # Sort by index
    sorted_idx = np.argsort(distances)
    # Exclude same point as unknown point
    if np.isclose(distances[0], 0):
        sorted_idx = sorted_idx[1:]
    return sorted_idx, distances

def calculate_nearest_neighbors_labels(X, Y, unknown_point, K):
    # Calculate and sort distances
    sorted_idx, distances = calculate_sorted_distances(X, unknown_point)
    # Get the first K labels (labels of K nearest neighbors)
    nearest_neighbors_labels = [Y[idx] for idx in sorted_idx[:K]]
    return nearest_neighbors_labels

def KNN_predict(X, Y, unknown_point, K):
    # Get KNN labels
    nearest_neighbors_labels = calculate_nearest_neighbors_labels(X, Y, unknown_point, K)
    # Create a set of classes with count per class in KNN labels
    classes, counts = np.unique(nearest_neighbors_labels, return_counts=True)
    # Get class(es) with the highest frequency
    max_count = np.max(counts)
    classes_with_max_count = classes[counts == max_count]
    # If one class has the highest frequency, return that class
    if len(classes_with_max_count) == 1:
        return classes_with_max_count[0]
    # Return closest neighbor's class in tiebreaker event
    else:
        return nearest_neighbors_labels[0]


def KNN_test(X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray, Y_test: np.ndarray, K: int):
    Y_label_predictions = []
    # Compile label predictions for each point
    for unknown_point in X_test:
        label_prediction = KNN_predict(X_train, Y_train, unknown_point, K)
        Y_label_predictions.append(label_prediction)

    # Compare label predictions (Y_label_predictions) to expected labels (Y_test)
    mask = Y_label_predictions == Y_test
    accuracy = sum(mask) / len(Y_test)
    return accuracy

