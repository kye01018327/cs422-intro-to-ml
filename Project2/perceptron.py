# %%
import numpy as np
from utils import load_data

# %%
# Test load data
num = 2
X, Y = load_data(f'data_{num}.txt')
print(X)
print(Y)

# %%
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

perceptron_train(X, Y)
# %%
def perceptron_test(X_test, Y_test, w, b):
    pass

