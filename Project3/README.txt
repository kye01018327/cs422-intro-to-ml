## distance_point_to_hyperplane(pt: np.ndarray, w: np.ndarray, b):

Calculates distance between point and hyperplane


## compute_margin(data: np.ndarray, w: np.ndarray, b):

Preprocess data
Iterate through points, calculate margin of each point and find the smallest margin


## generate_triplet_parameters(set1, set2, W: list, B: list, support_vectors: list):

Used to find w, b for three support vectors
For triplet with two positive, one negative
    Preprocess data
    Construct matrix to solve system of equations
    Calculate w, b by solving matrix

## svm_train_brute(training_data):

Separate data into two classes
Create unique pairs of the two sets
Calculate w, b for all pairs
Calculate unique triplets
Calculate w, b for all triplets
Validate the w, b of each combination of points
Compute the margin, keep the w, b of the largest margin

## svm_test_brute(w, b, x):

Calculate class of point using decision boundary

## plot_data_and_boundary(data, w, b):

Plot data points
Compute margin
Calculate and plot data and margin boundaries
Check if w[1] is close to zero (case for handling vertical lines)
    Calculate decision boundary
    Calculate the margin boundaries
    Create a range for the y-axis (x1) to plot vertical lines
    Plot decision boundary
    Plot margin boundary
Plot non-vertical lines
    Calculate decision boundary
    Calculate margin boundary
    Plot decision boundary
    Plot margin boundary
Miscellaneous settings

## svm_train_multiclass(training_data):

For each class
    Convert labels to +1 and -1 (for rest of classes)
    Train binary SVM one vs rest