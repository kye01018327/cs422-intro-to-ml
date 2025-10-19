import numpy as np

def perceptron_train(X: np.ndarray, Y: np.ndarray):
    # Initialize weights and bias
    w, b = np.zeros(X.shape[1]), 0

    # Until convergence
    epoch_limit = 500
    no_changes = False
    num_epochs = 1
    while not no_changes:
        no_changes = True
        for x, y in zip(X, Y):
            a = w @ x + b
            if y * a > 0:
                continue
            no_changes = False
            w = w + y * x
            b = b + y
        if no_changes:
            return w, b
        if num_epochs > epoch_limit:
            raise Exception('Exceeded epoch limit')
        num_epochs += 1

def perceptron_test(X_test: np.ndarray, Y_test: np.ndarray, w, b):
    predictions = [1 if w @ x + b > 0 else -1 for x in X_test]
    predictions = np.array(predictions)
    mask = predictions == Y_test
    accuracy = np.sum(mask) / Y_test.size
    return accuracy

