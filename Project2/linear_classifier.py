import numpy as np

def linear_train(X, Y, dLdw, dLdb, eta):
    w, b, max_iterations = np.zeros_like(X[0]), 0, 100
    for _ in range(max_iterations):
        total_grad_w = np.zeros_like(w)
        total_grad_b = 0
        for x, y in zip(X, Y):
            grad_w = dLdw(x, y, w, b)
            grad_b = dLdb(x, y, w, b)
            w -= eta * grad_w
            b -= eta * grad_b
            total_grad_w += grad_w
            total_grad_b += grad_b
        grad_norm = np.sqrt(np.sum(total_grad_w ** 2) + total_grad_b ** 2)
        if grad_norm < 1e-4:
            break

    return w, b

def linear_test(X_test: np.ndarray, Y_test: np.ndarray, w, b):
    predictions = [1 if w @ x + b > 0 else -1 for x in X_test]
    predictions = np.array(predictions)
    mask = predictions == Y_test
    accuracy = np.sum(mask) / Y_test.size
    return accuracy
