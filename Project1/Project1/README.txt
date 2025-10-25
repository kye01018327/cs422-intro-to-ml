decision_tree.py

--
calculate_entropy(Y):

1. Divide labels into classes and count per class
2. Calculate probabilities of classes (p_classes)
3. Calculate entropy as the sum of -p(c) * log(p(c)), where c is each element in c

--
calculate_information_gain(feature, labels):

1. Calculate entropy of labels
2. Create set of values and count per value of the feature
3. Calculate entropy of feature
    1. Iterate through each value in feature
    2. Calculate p(t), where t are the rows of the selected value
    3. Calculate H(t) or entropy of t
    4. Calculate p(t) * H(t) or weighted entropy of selected rows
4. Calculate information gain, where IG = Entropy of Labels - Sum[p(t) * H(t)], where t is each element in T, where T is the set of subsets (or selected rows) based on feature 
values

--
majority_class(Y):

1. Create set of classes and count per class
2. Return index of highest count
3. Return majority class (index of most frequent occurring class)

--
create_leaf(Y):

1. Return a dictionary with attributes
    {
        "type": "leaf",
        "result": majority_class(Y)
    }

--
DT_train_binary_helper(X, Y, remaining_features, depth):

1. Calculate the Entropy (H) for the current training set
2. Base Cases (Return a Leaf)
    1. Return leaf when:
        1. the entropy of labels is 0
        2. the maximum depth reached
        3. there are no more remaining features

3. Calculate information gain for each feature
4. Choose best feature based on highest information gain
5. If the best information gain is 0, return a leaf
6. Split feature matrix (X) into subsets (T)
    1. Create list ot store subsets (T)
    2. Select feature column with best information gain
    3. Iterate through best feature column's values
        1. Select rows based on selected value
        2. Construct training subset (t) with selected rows
        3. Append subset to list of subsets
7. Recursive Case (Create child nodes)
    1. Iterate through subsets
        1. Decrement depth counter to reflect going down one level in tree
        2. Don't decrement if depth is -1
        3. Remove current node's feature as an option for child nodes
        4. If subset has no rows (empty), create leaf
        5. Else create a child node recursively
8. Create parent node with attributes:
{
    "type": "node",
    "feature": best_feature
    "children": child_nodes
}

--
DT_train_binary(X, Y, max_depth):

1. Create array of feature indices
2. Create Decision Tree

--
DT_make_prediction(x, DT):

1. Base Cases
    1. Return result if node type is leaf
2. Recursive Cases
    1.Get the index of current node's feature
    2. Traverse child node based on value in feature row (x)

--
DT_test_binary(X, Y, DT):

1. Iterate through the rows of the feature matrix (X)
    1. Compute a prediction for each row using the decision tree (DT)
    2. Compile the prediction into a 1D array (predictions) with the same shape as the labels (Y)
2. Find the accuracy by comparing the predictions to the labels

---------------------------------------------

nearest_neighbors.py


--

calculate_sorted_distances(X, unknown_point):

1. Calculate distances between unknown point and known points
2. Sort by index
3. Exclude same point as unknown point

--
calculate_nearest_neighbors(X, Y, unknown_point, K):

1. Calculate and sort distances
2. Get the first K labels (labels of K nearest neighbors)

--
KNN_predict(X, Y, uknown_point, K):

1. Get KNN labels
2. Create a set of classes with count per class
3. Get class(es) with the highest frequency
4. If one class has the highest frequency, return that class
5. Return closest neighbor's class in tiebreaker event

--
KNN_test(X_train, Y_train, X_test, Y_test, K):

1. Compile label predictions for each point
2. Compare label predictions (Y_label_predictions) to expected labels (Y_test)

---------------------------------------------

clustering.py

--
randomize(X, K):

1. Fisher-Yates shuffle

--
recalculate_centroids(X, mu):

1. Create list of empty clusters
2. iterate through each point in the training set (X)
    1. Get distance between point and each centroid
    2. Select closest cluster (if same distance, select first occurrence or lowest index)
    3. Append point to the closest cluster
3. Create a new centroid
    1. Iterate through clusters
        1. If cluster has points, calculate new centroid for that cluster
        2. Else keep the old centroid

--
K_Means(X, K, mu):

1. If centroids (mu) are uninitialized, randomize starting points for K amount of clusters
2. Recalculate centroids until convergence


