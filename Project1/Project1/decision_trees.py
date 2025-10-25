# %%
import numpy as np
from math import log2

# %%
def calculate_entropy(Y):
    # Divide labels into classes and count per class
    classes, counts = np.unique(Y, return_counts=True)
    # Calculate probabilities of classes (p_classes)
    p_classes = counts / counts.sum()
    # Calculate entropy as the sum of -p(c) * log(p(c)), where c is each element in C
    entropy = np.sum([-p_c * log2(p_c) for p_c in p_classes])
    return entropy

# %%
def calculate_information_gain(feature, labels):
    # Calculate entropy of labels
    H_labels = calculate_entropy(labels)
    # Create set of values and count per value of the feature
    values, counts = np.unique(feature, return_counts=True)

    # Calculate entropy of feature
    H_feature = 0
    # Iterate through each value in feature
    for value, count in zip(values, counts):
        # Calculate p(t), where t are the rows of the selected value
        p_t = count / len(labels)
        # Calculate H(t) or entropy of t
        cls_labels = labels[feature == value]
        H_t = calculate_entropy(cls_labels)
        # Calculate p(t) * H(t)
        H_feature += p_t * H_t

    # Calculate information gain, where IG = Entropy of Labels - Sum[p(t) * H(t)], where t is each element in T, where T is the set of subsets (or selected rows) based on feature values
    ig = H_labels - H_feature
    return ig

# %%
def majority_class(Y):
    # Create set of classes and count per class
    classes, counts = np.unique(Y, return_counts=True)
    # Return index of highest count
    idx = np.argmax(counts)
    # Return majority class
    return classes[idx]

# %%
def create_leaf(Y):
    return {
        "type": "leaf",
        "result": majority_class(Y)
    }

# %%
def DT_train_binary_helper(X: np.ndarray, Y: np.ndarray, remaining_features: list, depth: int):
    entropy = calculate_entropy(Y) # Calculate the Entropy (H) for the current training set

    # Base Cases (Return a Leaf of majority class)
    # Return leaf when the:
    # Entropy of labels is 0
    if entropy == 0:
        return create_leaf(Y)
    
    # Maximum depth reached
    if depth == 0:
        return create_leaf(Y)
    
    # there are no more remaining features
    if len(remaining_features) == 0:
        return create_leaf(Y)

    # Calculate information gain for each feature
    igs_for_each_feature = []
    for feature_idx in remaining_features:
        feature = X[:, feature_idx]
        ig_for_this_feature = calculate_information_gain(feature, Y)
        igs_for_each_feature.append(ig_for_this_feature)
    
    # Choose best feature based on highest information gain
    best_feature_idx = np.argmax(igs_for_each_feature)
    best_feature = remaining_features[best_feature_idx]
    best_ig = igs_for_each_feature[best_feature_idx]

    # If the best information gain is 0, return a leaf
    if best_ig == 0:
        return create_leaf(Y)

    # Split feature matrix (X) into subsets (T)
    # Create list to store subsets (T)
    subsets = []
    # Select feature column with best information gain
    best_feature_col = X[:, best_feature]
    # Create set of feature values
    feature_values = np.unique(best_feature_col)
    # Iterate through best feature column's values
    for feature_value in feature_values:
        # Select rows based on selected value
        selected_rows = (best_feature_col == feature_value)
        # Construct training subset (t) with selected rows
        X_subset: np.ndarray = X[selected_rows]
        Y_subset = Y[selected_rows]
        # Append subset to list of subsets
        subsets.append((feature_value, X_subset, Y_subset))

    # Recursive Case (Create child nodes)
    child_nodes = {}
    # Iterate through subsets
    for feature_value, X_subset, Y_subset in subsets:
        # Decrement depth counter to reflect going down one level in tree
        new_depth = depth
        # Don't decrement if depth is -1
        if depth != -1:
            new_depth -= 1
        # Remove current node's feature as an option for child nodes
        child_remaining_features = [f for f in remaining_features if f != best_feature]
        # If subset has no rows (empty), create leaf
        if X_subset.shape[0] == 0:
            child_nodes[feature_value] = create_leaf(Y)
        # Else create a child node recursively
        else:
            child_nodes[feature_value] = DT_train_binary_helper(X_subset, Y_subset, child_remaining_features, new_depth)
    
    # Create and return parent node with attributes:
    node = {
        "type": "node",
        "feature": best_feature,
        "children": child_nodes
    }

    return node

def DT_train_binary(X: np.ndarray, Y, max_depth):
    # Create array of feature indices
    features = np.arange(X.shape[1])
    # Create Decision Tree
    return DT_train_binary_helper(X, Y, features, max_depth)

# %%
def DT_make_prediction(x: np.ndarray, DT: dict):
    node_type = DT['type']
    # Base Cases
    # Return result if node type is leaf
    if node_type == 'leaf':
        result = DT['result']
        return result

    # Recursive Cases
    # Get the index of current node's feature
    node_feature = DT['feature']
    # Traverse child node based on value in test feature row (x)
    selected_child = x[node_feature]
    return DT_make_prediction(x, DT['children'][selected_child])

# %%
def DT_test_binary(X: np.ndarray, Y: np.ndarray, DT: dict):
    # Iterate through the rows of the feature matrix (X)
    predictions = []
    for x in X:
        # Compute a prediction for each row using the decision tree (DT)
        row_prediction = DT_make_prediction(x, DT)
        # Compile the prediction into a 1D array (predictions) with the same shape as the labels (Y)
        predictions.append(row_prediction)

    # Find the accuracy by comparing the predictions to the labels
    predictions = np.array(predictions)
    mask = predictions == Y
    num_matching = np.sum(mask)
    accuracy = num_matching / len(Y)
    return accuracy

