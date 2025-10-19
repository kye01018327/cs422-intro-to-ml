# perceptron.py

## def perceptron_train(X, Y):

1. Initialize weights and bias
    1. Initialize weights vector to have same dimension as feature vector, all initial values as 0
    2. Initialize bias as a scalar with initial value of 0
2. Until convergence (no change over one epoch), train weights and bias
    1. For each feature vector in the feature dataset,
        1. Calculate the activation
            1. This being the product of the weight vector and feature vector, plus the bias
        2. Evaluate whether the activation aligns with the label (signed margin)
        3. If signed margin is negative (activation is opposite direction of label), adjust weights and bias
            1. if signed margin is negative (ya <= 0)
                1. w = w + yx
                2. b = b + y
                3. Track that a change occurred in this epoch
    2. If no change occurred in epoch, return weight vector and bias
    3. Keep track of number of epochs, if epochs exceed a limit, throw error

## def perceptron_test(X_test, Y_test, w, b)
    
1. Create a bool list representing the signed activation (prediction) of each feature vector
2. Find the sum of this bool list and divide by the feature vector length

# gradient_descent.py

## def gradient_descent(df, x_init, eta):

1. Initialize x to x_init
2. While magnitude of gradient is greater than 1e-4
    1. Calculate gradient
    2. Decrement x by the product of gradient and learning rate (eta)

# linear_classifier.py

## def linear_train(X, Y, dLdw, dLdb, eta):

1. While the magnitude of the combined gradient vector of w and b is greater than 1e-4
    1. Compute gradients
        1. grad_w = dLdw(...w)
        2. grad_b = dLdw(...b)
    2. w = w - eta * dLdw
    3. b = b - eta * dLdb
2. Return w, b

## def linear_test(X_test, Y_test, w, b):

1. Create a bool list representing the signed activation (prediction) of each feature vector
2. Find the sum of this bool list and divide by the feature vector length


