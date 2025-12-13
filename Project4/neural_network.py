# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
np.random.seed(0)
X, y = make_moons(200, noise=0.2)


# %%
def softmax(Z):
    Z_stable = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z_stable)
    sum_exp_Z = np.sum(exp_Z, axis=1, keepdims=True)
    Y_hat = exp_Z / sum_exp_Z
    return Y_hat


# %%
def calculate_loss(model: dict, X: np.ndarray, Y: np.ndarray):
    
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
def predict(model, x):
    a = x @ model['W1'] + model['b1']
    h = np.tanh(a)
    z = h @ model['W2'] + model['b2']
    y_hat = softmax(z)
    return np.argmax(y_hat)




# %%
print(len(y))
print(len(X[0]))



# %%
# Display data
plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)


# %%
