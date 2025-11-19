import numpy as np
import matplotlib.pyplot as plt

def distance_point_to_hyperplane(pt: np.ndarray, w: np.ndarray, b):
    dist = (w @ pt + b) / np.linalg.norm(w)
    return dist

def compute_margin(data: np.ndarray, w: np.ndarray, b):
    # Preprocess data
    X = data[:, :-1]
    Y = data[:, -1]

    # Iterate through points, calculate margin of each point and find the smallest margin
    min_margin = float('inf')
    for x, y in zip(X, Y):
        margin = y * distance_point_to_hyperplane(x, w, b)
        if margin < min_margin:
            min_margin = margin
    return min_margin

def generate_triplet_parameters(set1, set2, W: list, B: list, support_vectors: list):
    # For triplet with two positive, one negative
    for i in range(len(set1)):
        for j in range(i + 1, len(set1)):
            for k in range(len(set2)):
                # Preprocess data
                x1 = set1[i][:-1]
                x2 = set1[j][:-1]
                x3 = set2[k][:-1]

                # Construct matrix to solve system of equations
                A = np.vstack([
                    np.append(x1, 1),
                    np.append(x2, 1),
                    np.append(x3, 1)
                ])
                y = np.array([1, 1, -1])

                # Calculate w, b by solving matrix
                try:
                    solution = np.linalg.solve(A, y)
                    w = solution[:2]
                    b = solution[2]
                    W.append(w)
                    B.append(b)
                    support_vectors.append((set1[i], set1[j], set2[k]))
                except np.linalg.LinAlgError:
                    continue

def svm_train_brute(training_data):
    # Separate data into sets of two classes
    positive_set, negative_set = [], []
    for pt in training_data:
        if pt[-1] == 1:
            positive_set.append(pt)
        elif pt[-1] == -1:
            negative_set.append(pt)

    positive_set, negative_set = np.array(positive_set), np.array(negative_set)

    # Create unique pairs
    support_vectors = []
    pairs = []
    for pos in positive_set:
        for neg in negative_set:
            pairs.append((pos,neg))

    pairs = np.array(pairs)

    # Calculate w, b for all pairs
    W, B = [], []
    for pos, neg in pairs:
        x_pos, x_neg = pos[:-1], neg[:-1]
        d = x_pos - x_neg
        norm_d = np.linalg.norm(d)
        dir_w = d / norm_d

        w = dir_w * 2 / norm_d

        b = 1 - w @ x_pos
        W.append(w)
        B.append(b)
        support_vectors.append((pos, neg))
    
    # Create unique triplets
    # Calculate w, b for all triplets

    generate_triplet_parameters(positive_set, negative_set, W, B, support_vectors)
    generate_triplet_parameters(negative_set, positive_set, W, B, support_vectors)

    # Validate the w, b of each combination of points

    valid_W, valid_B, valid_S = [], [], []
    eps = 1e-8
    for w, b, s in zip(W, B, support_vectors):
        is_valid = True
        for point in training_data:
            x = point[:-1]
            y = point[-1]
            if y * (w @ x + b) < 1 - eps:
                is_valid = False
                break
        if is_valid:
            valid_W.append(w)
            valid_B.append(b)
            valid_S.append(s)

    # Compute margin, keep w, b, S with largest margin

    max_margin = 0
    for w, b, s in zip(valid_W, valid_B, valid_S):
        margin = compute_margin(training_data, w, b)
        if margin > max_margin:
            max_margin = margin
            w_db, b_db, S = w, b, s

    return w_db, b_db, np.array(S)

def svm_test_brute(w: np.ndarray, b, x: np.ndarray):
    y = w @ x + b
    if y > 0:
        return 1
    elif y <= 0:
        return -1

def plot_data_and_boundary(data, w, b):
    # Plot data points
    for item in data:
        if item[-1] == 1:
            plt.plot(item[0], item[1], 'b+')
        else:
            plt.plot(item[0], item[1], 'ro')

    # Compute margin

    margin = compute_margin(data, w, b)
    m = max(data[:, :-1].max(), abs(data[:, :-1].min())) + 1
    
    # Plot data boundary and margin
    
    # Check if w[1] is close to zero (case for handling vertical lines)
    if np.isclose(w[1], 0):
        # Calculate decision boundary
        x0_boundary = -b / w[0]
        
        # Calculate the margin boundaries
        norm_w = np.linalg.norm(w)
        x0_plus_margin = (-b - margin * norm_w) / w[0]
        x0_minus_margin = (-b + margin * norm_w) / w[0]
        
        # Create a range for the y-axis (x1) to plot vertical lines
        x1 = np.linspace(-m, m) 
        
        # Plot decision boundary
        plt.plot([x0_boundary, x0_boundary], [x1.min(), x1.max()], 
                 color='red', linewidth=2, label='Decision Boundary')
        
        # Plot margin boundary
        plt.plot([x0_plus_margin, x0_plus_margin], [x1.min(), x1.max()], 
                 color='orange', linestyle='--', label='Margin Boundary')
        plt.plot([x0_minus_margin, x0_minus_margin], [x1.min(), x1.max()], 
                 color='orange', linestyle='--')

    # Plot non vertical lines  
    else:
        x0 = np.linspace(-m, m)
        
        # Calculate decision boundary
        x1_boundary = (-w[0] * x0 - b) / w[1]

        # Calculate margin boundary
        x1_plus_margin = (-w[0] * x0 - (b - margin * np.linalg.norm(w))) / w[1]
        x1_minus_margin = (-w[0] * x0 - (b + margin * np.linalg.norm(w))) / w[1]

        # Plot decision boundary
        plt.plot(x0, x1_boundary, color='red', linewidth=2, label='Decision Boundary')
        
        # Plot margin boundary
        plt.plot(x0, x1_plus_margin, color='orange', linestyle='--', label='Margin Boundary')
        plt.plot(x0, x1_minus_margin, color='orange', linestyle='--')

    # Miscellaneous settings
    plt.axis([-m, m, -m, m])
    plt.xlabel('$x_0$')
    plt.ylabel('$x_1$')
    plt.title('Data, Boundary, and Margins')
    plt.legend()
    plt.grid(True)
    plt.show()
