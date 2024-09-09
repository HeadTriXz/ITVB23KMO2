import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix


# ==== OPGAVE 1 ====
def plot_number(nrVector: np.ndarray) -> None:
    """Plots the MNIST-image using Matplotlib.

    :param nrVector: The MNIST-image as a numpy array.
    """
    size = int(math.sqrt(nrVector.shape[0]))
    image = nrVector.reshape((size, size), order="F")

    plt.imshow(image, cmap="gray")
    plt.show()


# ==== OPGAVE 2a ====
def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns the sigmoid of the given value.

    :param z: The value to calculate the sigmoid of.
    :return: The sigmoid of the given value.
    """
    return 1 / (1 + np.exp(-z))


# ==== OPGAVE 2b ====
def get_y_matrix(y: np.ndarray, m: int) -> np.ndarray:
    """
    Returns a sparse matrix with a 1 on the position of y_i and a 0 on the other positions.

    :param y: The vector with values y_i from 1...x.
    :param m: The number of rows in the matrix.
    :return: A sparse matrix of m × x with a 1 on position y_i and a 0 on the other positions.
    """
    rows = np.arange(m)
    data = np.ones(m)

    mat = csr_matrix((data, (rows, y[:, 0] - 1)))
    return mat.toarray()


# ==== OPGAVE 2c ====
# ===== deel 1: =====
def predict_number(Theta2: np.ndarray, Theta3: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Predicts the number of a MNIST-image.

    :param Theta2: The weights of the second layer.
    :param Theta3: The weights of the third layer.
    :param X: The MNIST-image.
    :return: The predicted number of the MNIST-image.
    """
    a1 = np.insert(X, 0, 1, axis=1)

    a2 = sigmoid(a1 @ Theta2.T)
    a2 = np.insert(a2, 0, 1, axis=1)

    return sigmoid(a2 @ Theta3.T)


# ===== deel 2: =====
def compute_cost(
        Theta2: np.ndarray,
        Theta3: np.ndarray,
        X: np.ndarray,
        y: np.ndarray
) -> float:
    """
    Computes the cost of the prediction.

    :param Theta2: The weights of the second layer.
    :param Theta3: The weights of the third layer.
    :param X: The MNIST-images.
    :param y: The actual numbers of the MNIST-images.
    :return: The cost of the prediction.
    """
    m = X.shape[0]
    y_mat = get_y_matrix(y, m)

    # Predict the numbers
    predictions = predict_number(Theta2, Theta3, X)

    # Calculate the cost
    j = np.sum(-y_mat * np.log(predictions) - (1 - y_mat) * np.log(1 - predictions))

    return j / m


# ==== OPGAVE 3a ====
def sigmoid_gradient(z):
    """
    Returns the gradient of the sigmoid function.

    :param z: The value to calculate the gradient of the sigmoid of.
    :return: The gradient of the sigmoid of the given value.
    """
    return sigmoid(z) * (1 - sigmoid(z))


# ==== OPGAVE 3b ====
def nn_check_gradients(
        Theta2: np.ndarray,
        Theta3: np.ndarray,
        X: np.ndarray,
        y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the gradients of the weights using the backpropagation algorithm.

    :param Theta2: The weights of the second layer.
    :param Theta3: The weights of the third layer.
    :param X: The MNIST-images.
    :param y: The actual numbers of the MNIST-images.
    :return: The gradients of the weights.
    """
    m = X.shape[0]
    y_mat = get_y_matrix(y, m)

    # Copied from 'predict_number'
    a1 = np.insert(X, 0, 1, axis=1)

    a2 = sigmoid(a1 @ Theta2.T)
    a2 = np.insert(a2, 0, 1, axis=1)

    a3 = sigmoid(a2 @ Theta3.T)

    # Perform backpropagation
    Delta2 = np.zeros(Theta2.shape)
    Delta3 = np.zeros(Theta3.shape)

    for i in range(m):
        d3 = a3[i] - y_mat[i]
        d2 = Theta3.T @ d3 * sigmoid_gradient(a2[i])

        Delta2 += np.outer(d2[1:], a1[i])
        Delta3 += np.outer(d3, a2[i])

    Delta2_grad = Delta2 / m
    Delta3_grad = Delta3 / m

    return Delta2_grad, Delta3_grad
