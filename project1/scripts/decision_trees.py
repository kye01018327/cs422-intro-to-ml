# %%
import sys
sys.path.append('..')


# %%
import numpy as np
from math import log2
from utils import load_data

# %%
from pprint import pprint

# %%
def calculate_entropy(Y):
    classes, counts = np.unique(Y, return_counts=True)
    p_C = counts / counts.sum()
    calculate_entropy = np.sum([-p_c * log2(p_c) for p_c in p_C])
    return calculate_entropy

# %%
def calculate_information_gain(feature, labels):
    # Calculate entropy of labels
    H_labels = calculate_entropy(labels)
    # Create set of classes and counts for feature
    classes, counts = np.unique(feature, return_counts=True)

    # Calculate entropy of feature
    H_feature = 0
    # Iterate through each c in set C
    for cls, count in zip(classes, counts):
        # Calculate p(t)
        p_t = count / len(labels)
        # Calculate H(t) or entropy of c
        cls_labels = labels[feature == cls]
        H_t = calculate_entropy(cls_labels)
        # Calculate p(t) * H(t)
        H_feature += p_t * H_t

    # Information Gain = Entropy of Labels - Sum[p(t) * H(t)] where c in C
    ig = H_labels - H_feature
    return ig

# %%
def majority_class(Y):
    # Create set of classes, counts for each class
    classes, counts = np.unique(Y, return_counts=True)
    # Return index of highest count
    idx = np.argmax(counts)
    # Return majority class
    return classes[idx].item()

# %%
def create_leaf(Y):
    return {
        "type": "leaf",
        "result": majority_class(Y)
    }

# %%
def DT_train_binary_helper(X: np.ndarray, Y: np.ndarray, remaining_features: list, depth: int):
    # print('SUBTREE----------------')
    # print(f'X: {X}')
    # print(f'Y: {Y}')
    # print(f'remaining_features: {remaining_features}')
    # print(f'depth: {depth}')
    entropy = calculate_entropy(Y) # Calculate the Entropy (H) for the entire training set
    # print(f'entropy: {entropy}')
    # Base Cases (Return a Leaf)
    if entropy == 0:
        # print('BASE CASE ENTROPY')
        return create_leaf(Y)
    
    if depth == 0:
        # print('BASE CASE DEPTH')
        return create_leaf(Y)
    
    if len(remaining_features) == 0:
        # print('BASE CASE NO REMAINING FEATURES')
        return create_leaf(Y)

    # Calculate Information Gain for each split
    igs_for_each_feature = []
    for feature_idx in remaining_features:
        feature = X[:, feature_idx]
        ig_for_this_feature = calculate_information_gain(feature, Y)
        igs_for_each_feature.append(ig_for_this_feature)
    
    # print(f'igs_for_each_feature: {igs_for_each_feature}')
    # Choose to split on the feature that gives the best IG
    best_feature_idx = np.argmax(igs_for_each_feature)
    best_feature = remaining_features[best_feature_idx]
    best_ig = igs_for_each_feature[best_feature_idx]

    # print(f'best_feature: {best_feature}')
    # print(f'best_ig: {best_ig}')

    if best_ig == 0:
        # print('BASE CASE BEST IG == 0')
        return create_leaf(Y)

    # Create splits
    splits = []
    best_feature_col = X[:, best_feature]
    # print(f'best_feature_col: {best_feature_col}')
    feature_classes = np.unique(best_feature_col)
    # print(f'feature_classes: {feature_classes}')
    for cls in feature_classes:
        selected_rows = (best_feature_col == cls)
        X_child: np.ndarray = X[selected_rows]
        Y_child = Y[selected_rows]
        splits.append((cls, X_child, Y_child))

    # print('splits: ', end='')
    # pprint(splits)

    # Create child nodes
    nodes = {}
    for cls, X_child, Y_child in splits:
        new_depth = depth
        if depth != -1:
            new_depth -= 1
        child_remaining_features = [int(f) for f in remaining_features if f != best_feature]
        
        if X_child.shape[0] == 0:
            nodes[cls.item()] = create_leaf(Y)
        else:
            nodes[cls.item()] = DT_train_binary_helper(X_child, Y_child, child_remaining_features, new_depth)
    
    # Create node
    node = {
        "type": "node",
        "feature": int(best_feature),
        "children": nodes
    }

    return node

def DT_train_binary(X: np.ndarray, Y, max_depth):
    # Create array of feature indexes
    features = np.arange(X.shape[1])
    return DT_train_binary_helper(X, Y, features, max_depth)

# %%
def DT_make_prediction(x: np.ndarray, DT: dict):
    node_type = DT['type']
    # Base Cases
    if node_type == 'leaf':
        result = DT['result']
        # print(result)
        return result

    # Recursive Cases
    # pprint(DT)

    node_feature = DT['feature']
    selected_child = x[node_feature]
    return DT_make_prediction(x, DT['children'][selected_child])

# %%
def DT_test_binary(X: np.ndarray, Y: np.ndarray, DT: dict):
    # Iterate through the rows of the feature set (X)
    predictions = []
    for x in X:
        # Compute a prediction for each row using the decision tree (DT)
        row_prediction = DT_make_prediction(x, DT)
        # Compile the prediction into a 1D set (predictions) with the same shape as the labels (Y)
        predictions.append(row_prediction)

    # Find the accuracy by comparing the predictions to the labels
    predictions = np.array(predictions)
    intersection = predictions == Y
    num_matching = np.sum(intersection)
    accuracy = num_matching / len(Y)
    return accuracy
