import numpy as np

def distance_point_to_hyperplane(pt: np.ndarray, w: np.ndarray, b):
    dist = np.abs(w @ pt + b) / np.linalg.norm(w)
    return dist

def compute_margin(data: np.ndarray, w: np.ndarray, b):
    X = data[:, :-1]
    Y = data[:, -1]
    distances = w @ X + b / np.linalg.norm(w)
    d_positive = np.min(distances[distances > 0]) if np.any(distances > 0) else 0
    d_negative = np.max(distances[distances < 0]) if np.any(distances < 0) else 0

    margin = d_positive + np.abs(d_negative)
    return margin


def svm_train_brute(training_data):
    w,b,S = 0,0,[]

    return w,b,S