import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm

# OPGAVE 1
def draw_graph(data: np.ndarray) -> None:
    """
    Draw a scatter plot of the given data.

    :param data: The data to plot.
    """
    plt.scatter(data[:, 0], data[:, 1])
    plt.xlabel("Population in 10,000s")
    plt.ylabel("Profit in $10,000s")
    plt.show()


# OPGAVE 2
def compute_cost(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    """
    Compute the cost of the current values of theta.

    :param X: The features.
    :param y: The actual values.
    :param theta: The weights of the model.
    :return: The cost of the prediction.
    """
    m = len(y)
    predictions = np.dot(X, theta)
    errors = np.power(predictions - y, 2)

    return np.sum(errors) / (2 * m)


# OPGAVE 3a
def gradient_descent(
        X: np.ndarray,
        y: np.ndarray,
        theta: np.ndarray,
        alpha: float,
        num_iters: int
) -> tuple[np.ndarray, list[float]]:
    """
    Perform gradient descent to find the optimal values for theta.

    :param X: The features.
    :param y: The actual values.
    :param theta: The initial weights of the model.
    :param alpha: The learning rate.
    :param num_iters: The number of iterations.
    :return: The optimal weights of the model and the costs during training.
    """
    m, n = X.shape
    costs = []

    for i in range(1, num_iters):
        predictions = np.dot(X, theta.T)
        errors = predictions - y

        theta -= (alpha / m) * np.sum(errors * X, axis=0)

        costs.append(compute_cost(X, y, theta.T))

    return theta, costs


# OPGAVE 3b
def draw_costs(data: list[float]) -> None:
    """
    Plot the costs during training.

    :param data: The costs during training.
    """
    plt.plot(data)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.show()


# OPGAVE 4
def contour_plot(X: np.ndarray, y: np.ndarray) -> None:
    """
    Draw a contour plot for different values of theta_0 and theta_1.

    :param X: The features.
    :param y: The actual values.
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    t1 = np.linspace(-10, 10, 100)
    t2 = np.linspace(-1, 4, 100)
    T1, T2 = np.meshgrid(t1, t2)

    J_vals = np.zeros((len(t2), len(t2)))

    for i in range(len(t1)):
        for j in range(len(t2)):
            theta = np.array([t1[i], t2[j]]).reshape(2, 1)
            J_vals[i, j] = compute_cost(X, y, theta)

    ax.plot_surface(T1, T2, J_vals, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    ax.set_xlabel(r'$\theta_0$', linespacing=3.2)
    ax.set_ylabel(r'$\theta_1$', linespacing=3.1)
    ax.set_zlabel(r'$J(\theta_0, \theta_1)$', linespacing=3.4)

    ax.dist = 10

    plt.show()
