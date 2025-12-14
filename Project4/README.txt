firstname_lastname_project4.zip
kevin_ye_project4.zip

def softmax(Z):

    This function computes the softmax of Z - the matrix of the forward passes based off the samples, outputing Y_hat which is softmax(Z)

    Parameters
    ----------
    Z: np.ndarray
        Array of (N,2) shape, represents z in forward pass, consisting of those results of (1,2) shape N samples
    
    Returns
    -------
    Y_hat: np.ndarray
        Array of (N,2) shape, the prediction for each sample
    
    Process
    -------
    1. Calculate the matrix of e^zj with shape (N, 2)
    2. Calculate the sum of e^z for each sample, resulting array should have (N, 1) shape
    3. Find the quotient of the two matrices
    4. Return quotient as Y_hat
    

def calculate_loss(model: dict, X: np.ndarray, Y: np.ndarray):

    This function calculates the loss according to the assignemnt document.

    Parameters
    ----------
    model: dict
        has structure
        {
            'W1': matrix of shape (2, nn_hdim),
            'W2': matrix of shape (nn_hdim, 2),
            'b1': matrix of shape (1, nn_hdim),
            'b2': matrix of shape (1, 2)
        }
        representing the model parameters
    
    Returns
    -------
    loss: float
        The loss of the batch
    
    Process
    -------
    1. Calculate the prediction matrix
    2. Calculate the loss according to the assignment document
        1. Calculate the inner sum for each sample
        2. Sum the matrix then divide by N and negate
    3. Return loss


def predict(model:, x: np.ndarray):
    
    This function does a forward pass then returns the majority class of the output

    Parameters
    ----------
    model: dict
        same as before
    
    x: np.ndarray
        shape (1,2), takes one sample for forward pass

    Returns
    -------
    y_hat: int
        Either 0 or 1 (the class)

    Process
    -------
    1. Forward pass, then select index with highest value
    2. Return result as prediction y_hat


def adjust_y_shape(Y: np.ndarray):

    This function just changes the shape of Y from (N,) to (N, 2) for broadcasting compatibility

    Parameters
    ----------
    Y: np.ndarray
        matrix of labels for each sample (200,) shape

    Returns
    -------
    Y: np.ndarray
        same matrix but adjusted to be (200, 2) shape


def build_model(X: np.ndarray, Y: np.ndarray, nn_hdim, num_passes=20000, print_loss=False):

    This function builds the model according to the assignment requirements

    Parameters
    ----------
    X: np.ndarray
        samples with shape (N, 2)
    
    Y: np.ndarray
        samples with shape (N,)

    nn_hdim: int
        specifies number of hidden nodes in hidden layer

    num_passes: int
        specifies number of passes or epochs

    print_loss: bool
        if True, prints initial loss, and loss every 1000 iterations

    Returns
    -------
    model: dict
        in the same format as before

    Process
    -------
    1. Adjust the shape of matrix Y to be compatible with calculations Y - Y_hat
    2. Initialize weights and biases
    3. Iterate for specified num_passes
        1. Every 1000 iterations print loss if enabled
        2. Calculate gradients one to one according to assignment document
        3. Update model parameters by gradient descent
    4. Return model

