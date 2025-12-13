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
def calculate_loss(model: dict, X, y):
    N = len(X)
    pass


# %%
print(X)
print(y)


# %%
# Display data
plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
