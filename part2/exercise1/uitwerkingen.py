import numpy as np
from sklearn.datasets import load_iris


# 1. Load the iris dataset.
dataset = load_iris()

# 2. Extract the features.
X = dataset.data

# 3. Extract the target and convert it to binary.
y = (dataset.target == 2).astype(int)

# 4. Define the sigmoid function.
def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns the sigmoid of the given value.

    :param z: The value to calculate the sigmoid of.
    :return: The sigmoid of the given value.
    """
    return 1 / (1 + np.exp(-z))

# 5. Initialize the theta values.
theta = np.ones(X.shape[1])

# 6. Define the cost function.
def compute_cost(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """
    Computes the cost of the prediction.

    :param theta: The weights of the model.
    :param X: The features.
    :param y: The actual values.
    :return: The cost of the prediction.
    """
    m = X.shape[0]
    predictions = sigmoid(X @ theta)

    # Calculate the cost
    j = np.sum(-y * np.log(predictions) - (1 - y) * np.log(1 - predictions))

    return j / m

# 7. Define the gradient function.
def compute_gradient(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Computes the gradient of the cost function.

    :param theta: The weights of the model.
    :param X: The features.
    :param y: The actual values.
    :return: The gradient of the cost function.
    """
    m = X.shape[0]
    predictions = sigmoid(X @ theta)

    return X.T @ (predictions - y) / m

# 8. Define the gradient descent function.
def gradient_descent(
        theta: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        alpha: float,
        iterations: int,
        debug: bool = False
) -> np.ndarray:
    """
    Performs gradient descent.

    :param theta: The initial weights of the model.
    :param X: The features.
    :param y: The actual values.
    :param alpha: The learning rate.
    :param iterations: The number of iterations.
    :param debug: Whether to print the cost during training.
    :return: The optimal weights of the model.
    """
    for i in range(iterations):
        theta -= alpha * compute_gradient(theta, X, y)

        # Visualize the progress made during training.
        if debug and i % 100 == 0:
            print(f"[{i}]: cost = {compute_cost(theta, X, y)}")

    return theta

# 9. Perform gradient descent.
print("=========== Training ===========")
theta = gradient_descent(theta, X, y, 0.01, 1500, debug=True)

# 10. Print the cost of the prediction.
final_cost = compute_cost(theta, X, y)
print("=========== Results ===========")
print(f"Final cost: {final_cost}")

# 11. Experiment with different learning rates and iterations.
print("=========== Experiment ===========")
theta = np.ones(X.shape[1])
theta = gradient_descent(theta, X, y, 0.01, 5000)
print(f"Final cost (alpha=0.01, iterations=5000): {compute_cost(theta, X, y)}")

theta = np.ones(X.shape[1])
theta = gradient_descent(theta, X, y, 0.1, 2000)
print(f"Final cost (alpha=0.1, iterations=2000): {compute_cost(theta, X, y)}")

theta = np.ones(X.shape[1])
theta = gradient_descent(theta, X, y, 0.5, 1000)
print(f"Final cost (alpha=0.5, iterations=1000): {compute_cost(theta, X, y)}")
