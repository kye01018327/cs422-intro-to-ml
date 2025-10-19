import numpy as np

def gradient_descent(df, x_init, eta):
    x = x_init
    gradient = df(x)
    while np.sqrt(np.sum(np.square(gradient))) > 1e-4:
       x = x - eta * gradient
       gradient = df(x)

    return x


