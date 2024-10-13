import numpy as np


def add_temperature(predictions, temperature):
    """
    Add a temperature to the predictions. The higher the temperature, the more uncertain the predictions will be.

    :param predictions: The predictions to add the temperature to.
    :param temperature: The temperature to add.
    :return: The predictions with the temperature added.
    """
    predictions = np.log(predictions) / temperature
    exp_preds = np.exp(predictions)

    return exp_preds / np.sum(exp_preds)


def get_vectors(embeddings, words):
    """
    Input:
        embeddings: a word 
        fr_embeddings:
        words: a list of words
    Output: 
        X: a matrix where the rows are the embeddings corresponding to the rows on the list
        
    """
    m = len(words)
    X = np.zeros((1, 300))
    for word in words:
        english = word
        eng_emb = embeddings[english]
        X = np.row_stack((X, eng_emb))
    X = X[1:,:]
    return X
