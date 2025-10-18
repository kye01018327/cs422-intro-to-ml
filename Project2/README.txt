# perceptron.py

## def perceptron_train(X, Y):

# Initialize weights and bias
    # Initialize weights vector to have same dimension as feature vector, all initial values as 0
    # Initialize bias as a scalar with initial value of 0
# Until convergence (no change over one epoch), train weights and bias
    # For each feature vector in the feature dataset,
        # Calculate the activation
            # This being the sum of the product of the weight vector and feature vector, plus the bias
        # Evaluate whether the activation aligns with the label (signed margin)
        # If signed margin is negative (activation is opposite direction of label), adjust weights and bias
            # if signed margin is negative (ya <= 0)
                # w = w + yx
                # b = b + y
                # Track that a change occurred in this epoch
    # If no change occurred in epoch, return weight vector and bias
    # Keep track of number of epochs, if epochs exceed a limit, throw error