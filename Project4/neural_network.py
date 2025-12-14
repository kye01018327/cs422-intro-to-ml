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
    return np.argmax(y_hat, axis=1)


def fix_y_dimensions(Y: np.ndarray):
    N = Y.shape[0]
    result = np.zeros((N, 2))
    result[np.arange(N), Y] = 1
    return result


# %%
def build_model(X: np.ndarray, Y: np.ndarray, nn_hdim, num_passes=20000, print_loss=False):
    def parameters(model, X):
        A = X @ model['W1'] + model['b1']
        H = np.tanh(A)
        Z = H @ model['W2'] + model['b2']
        Y_hat = softmax(Z)
        return A,H,Z,Y_hat

    Y = fix_y_dimensions(Y)
    learning_rate = 0.01
    # Initialize weights and biases
    model = {}
    model['W1'] = np.random.randn(2,nn_hdim) * 0.1
    model['W2'] = np.random.randn(nn_hdim,2) * 0.1
    model['b1'] = np.zeros((1,nn_hdim))
    model['b2'] = np.zeros((1,2))

    # Calculate gradients
    for i in range(num_passes):
        if i % 1000 == 0 and print_loss:
            print(calculate_loss(model, X, Y))
        A, H, Z, Y_hat = parameters(model, X)
        dL_dy_hat = Y_hat - Y
        dL_da = (1 - np.tanh(A)**2) * (dL_dy_hat @ model['W2'].T)
        dL_dW2 = H.T @ dL_dy_hat
        dL_db2 = np.sum(dL_dy_hat, axis=0, keepdims=True)
        dL_dW1 = X.T @ dL_da
        dL_db1 = np.sum(dL_da, axis=0, keepdims=True)

        # Update weights and bias
        model['W1'] -= learning_rate * dL_dW1
        model['W2'] -= learning_rate * dL_dW2
        model['b1'] -= learning_rate * dL_db1
        model['b2'] -= learning_rate * dL_db2

    return model


# %%
# Display data
plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)


# %%

def plot_decision_boundary(pred_func, X, y):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)


# %%
plt.figure(figsize=(16,32))
hidden_layer_dimensions = [1, 2, 3, 4]
for i, nn_hdim in enumerate(hidden_layer_dimensions):
    plt.subplot(5, 2, i+1)
    plt.title('HiddenLayerSize%d' % nn_hdim)
    model = build_model(X, y, nn_hdim)
    plot_decision_boundary(lambda x: predict(model, x), X, y)
plt.show()

# %%
