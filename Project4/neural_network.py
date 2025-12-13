# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
np.random.seed(0)
X, y = make_moons(200, noise=0.2)


# Helper function to evaluate the total loss on the dataset
# model is the current version of the model
# {
#     'W1': W1,
#     'b1': b1,
#     'W2': W2,
#     'b2': b2
# }
# X is all the training labels
# y is the training labels

# %%
def calculate_loss(model: dict, X: np.ndarray, Y: np.ndarray):
    def softmax(Z):
        Z_stable = Z - np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(Z_stable)
        sum_exp_Z = np.sum(exp_Z, axis=1, keepdims=True)
        Y_hat = exp_Z / sum_exp_Z
        return Y_hat
    
    # Calculate prediction matrix
    A = X @ model['W1'] + model['b1']
    H = np.tanh(A)
    Z = H @ model['W2'] + model['b2']
    Y_hat = softmax(Z)

    N = len(X)
    epsilon = 1e-12
    loss = np.sum(Y * np.log(Y_hat + epsilon), axis=1)
    loss = np.sum(loss)
    loss = -loss / N
    return loss


# %%
print(len(y))
print(len(X[0]))



# %%
# Display data
plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)


# %%
