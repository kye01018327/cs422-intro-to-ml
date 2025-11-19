import numpy as np
import matplotlib.pyplot as plt

def distance_point_to_hyperplane(pt: np.ndarray, w: np.ndarray, b):
    dist = np.abs(w @ pt + b) / np.linalg.norm(w)
    return dist

def compute_margin(data: np.ndarray, w: np.ndarray, b):
    X = data[:, :-1]
    Y = data[:, -1]
    min_margin = float('inf')
    for x, y in zip(X, Y):
        margin = y * (w @ x + b) / np.linalg.norm(w)
        if margin < min_margin:
            min_margin = margin
    return min_margin

def generate_triplet_parameters(set1, set2, W: list, B: list, support_vectors: list):
    for i in range(len(set1)):
        for j in range(i + 1, len(set1)):
            for k in range(len(set2)):
                x1 = set1[i][:-1]
                x2 = set1[j][:-1]
                x3 = set2[k][:-1]
                A = np.vstack([
                    np.append(x1, 1),
                    np.append(x2, 1),
                    np.append(x3, 1)
                ])
                y = np.array([1, 1, -1])
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
    # Partition data by class
    positive_set, negative_set = [], []
    for pt in training_data:
        if pt[-1] == 1:
            positive_set.append(pt)
        elif pt[-1] == -1:
            negative_set.append(pt)

    positive_set, negative_set = np.array(positive_set), np.array(negative_set)

    # Enumerate all possible support-vector sets
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
    
    generate_triplet_parameters(positive_set, negative_set, W, B, support_vectors)
    generate_triplet_parameters(negative_set, positive_set, W, B, support_vectors)

    # Validate each candidate (w, b)

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

    valid_W = np.array(valid_W)
    valid_B = np.array(valid_B)
    valid_S = np.array(valid_S)

    # Compute margin, keep (w, b) with largest margin

    max_margin = 0
    min_idx = None
    for w, b, s in zip(valid_W, valid_B, valid_S):
        margin = compute_margin(training_data, w, b)
        if margin > max_margin:
            max_margin = margin
            w_db, b_db, S = w, b, s

    return w_db, b_db, S

def plot_data_and_boundary(data, w, b):
    # --- Plotting the Data Points (No Change) ---
    for item in data:
        if item[-1] == 1:
            plt.plot(item[0], item[1], 'b+')
        else:
            plt.plot(item[0], item[1], 'ro')

    min_margin = compute_margin(data, w, b)
    m = max(data[:, :-1].max(), abs(data[:, :-1].min())) + 1
    
    # --- Plotting the Decision Boundary and Margins (Corrected) ---
    
    # Check if w[1] is close to zero (indicating a near-vertical line)
    if np.isclose(w[1], 0):
        # The boundary is x0 = -b / w[0] (a vertical line)
        x0_boundary = -b / w[0]
        
        # Calculate the margin boundaries
        # x0_plus_margin = (-b + min_margin * np.linalg.norm(w)) / w[0]
        # x0_minus_margin = (-b - min_margin * np.linalg.norm(w)) / w[0]
        # The equation for the margin boundaries are: w[0]x0 + b = +- min_margin * ||w||
        
        norm_w = np.linalg.norm(w)
        x0_plus_margin = (-b - min_margin * norm_w) / w[0]
        x0_minus_margin = (-b + min_margin * norm_w) / w[0]
        
        # Create a range for the y-axis (x1) to plot the vertical lines
        x1 = np.linspace(-m, m) 
        
        # Decision Boundary
        # plt.plot expects x, y. For a vertical line, x is constant, y is a range.
        plt.plot([x0_boundary, x0_boundary], [x1.min(), x1.max()], 
                 color='red', linewidth=2, label='Decision Boundary')
        
        # Margin Boundaries
        plt.plot([x0_plus_margin, x0_plus_margin], [x1.min(), x1.max()], 
                 color='orange', linestyle='--', label='Margin Boundary')
        plt.plot([x0_minus_margin, x0_minus_margin], [x1.min(), x1.max()], 
                 color='orange', linestyle='--')
                 
    else:
        # Original logic for non-vertical lines (where x1 is a function of x0)
        x0 = np.linspace(-m, m)
        
        # The original calculations for x1 as a function of x0 (for horizontal and diagonal lines)
        x1_boundary = (-w[0] * x0 - b) / w[1]
        x1_plus_margin = (-w[0] * x0 - (b - min_margin * np.linalg.norm(w))) / w[1]
        x1_minus_margin = (-w[0] * x0 - (b + min_margin * np.linalg.norm(w))) / w[1]

        # Plot the lines
        plt.plot(x0, x1_boundary, color='red', linewidth=2, label='Decision Boundary')
        plt.plot(x0, x1_plus_margin, color='orange', linestyle='--', label='Margin Boundary')
        plt.plot(x0, x1_minus_margin, color='orange', linestyle='--')

    # --- Final Plot Settings (No Change) ---
    plt.axis([-m, m, -m, m])
    plt.xlabel('$x_0$')
    plt.ylabel('$x_1$')
    plt.title('Data, Boundary, and Margins')
    plt.legend()
    plt.grid(True)
    plt.show()

# [Image of a 2D scatter plot showing a decision boundary and two parallel margin lines, 
# with the decision boundary being vertical, separating two different classes of data points]
