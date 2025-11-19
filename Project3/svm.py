import numpy as np
import matplotlib.pyplot as plt

def distance_point_to_hyperplane(pt: np.ndarray, w: np.ndarray, b):
    dist = np.abs(w @ pt + b) / np.linalg.norm(w)
    return dist

def compute_margin(data: np.ndarray, w: np.ndarray, b):
    X = data[:, :-1]
    min_dist_to_boundary = float('inf')
    for pt in X:
        dist = distance_point_to_hyperplane(pt, w, b) 
        if dist < min_dist_to_boundary:
            min_dist_to_boundary = dist
    margin = 2 * min_dist_to_boundary
    return margin

def svm_train_brute(training_data: np.ndarray):
    # Preprocessing
    # Separate into classes
    positive_set = []
    negative_set = []
    for pt in training_data:
        label = pt[-1]
        if label == 1:
            positive_set.append(pt)
        elif label == -1:
            negative_set.append(pt)

    positive_set = np.array(positive_set)
    negative_set = np.array(negative_set)

    # Enumerate all possible support-vector combinations
    # 2 support vectors

    two_sv = []
    for pos_pt in positive_set:
        for neg_pt in negative_set:
            two_sv.append((pos_pt, neg_pt))
    two_sv = np.array(two_sv)

    # Compute w and b for each pair of support vectors
    for pair in two_sv:
        positive_pt = pair[0][:-1]
        negative_pt = pair[1][:-1]
        dist = positive_pt - negative_pt
        dir_w = dist / np.linalg.norm(dist)
        gamma = np.linalg.norm(dist / 2)
        this_w = dir_w / gamma
        b = 1 - this_w @ positive_pt

    # Separate into classes

    # Enumerate all possible support-vector combinations
    # 2 support vectors
    # 3 support vectors
    pass

def plot_data_and_boundary(data, w, b):
    """
    Plots 2D data points and the SVM decision boundary.
    
    Parameters:
        data : numpy array of shape (N, 3)
               Each row: [x1, x2, y], where y in {-1, 1}
        w    : numpy array of shape (2,), normal vector of boundary
        b    : float, bias term
    """

    # 1. Separate positive and negative points
    pos = data[data[:, 2] == 1]
    neg = data[data[:, 2] == -1]

    # 2. Plot points
    plt.scatter(pos[:, 0], pos[:, 1], color='blue', label='Positive (+1)')
    plt.scatter(neg[:, 0], neg[:, 1], color='red', label='Negative (-1)')

    # 3. Plot decision boundary
    # Decision boundary: w1*x1 + w2*x2 + b = 0
    x_vals = np.linspace(min(data[:,0])-1, max(data[:,0])+1, 200)
    if w[1] != 0:
        # Solve for x2
        y_vals = -(w[0] * x_vals + b) / w[1]
    else:
        # Vertical line case
        x_vals = np.full(200, -b / w[0])
        y_vals = np.linspace(min(data[:,1])-1, max(data[:,1])+1, 200)

    plt.plot(x_vals, y_vals, 'k-', label='Decision boundary')

    # 4. Plot margins (optional)
    if w[1] != 0:
        margin = 1 / np.linalg.norm(w)
        plt.plot(x_vals, y_vals + margin, 'k--', label='Margin')
        plt.plot(x_vals, y_vals - margin, 'k--')
    else:
        margin = 1 / np.linalg.norm(w)
        plt.axvline(x=-b/w[0] + margin, linestyle='--', color='k')
        plt.axvline(x=-b/w[0] - margin, linestyle='--', color='k')

    # 5. Labels & legend
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('SVM Decision Boundary')
    plt.legend()
    plt.axis('equal')
    plt.show()
