import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras


# OPGAVE 1a
def plot_image(img: np.ndarray, label: str) -> None:
    """
    Plot the image with the given label.

    :param img: The image to plot.
    :param label: The label of the image.
    """
    plt.imshow(img, cmap=plt.cm.binary)
    plt.title(label)
    plt.show()


# OPGAVE 1b
def scale_data(X: np.ndarray) -> np.ndarray:
    """
    Scale the data to be between 0 and 1.

    :param X: The data to scale.
    :return: The scaled data.
    """
    return X / np.max(X)

# OPGAVE 1c
def build_model():
    """
    Initialize the model.

    :return: The model.
    """
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def save_model(model: keras.Model) -> None:
    """
    Save the model to a file.

    :param model: The model to save.
    """
    model.save("../../models/model.keras")
