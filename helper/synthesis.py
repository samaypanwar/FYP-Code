import numpy as np


def create_features_linspace(vector_max: np.array, vector_min: np.array, num: int) -> np.array:
    """This function creates the features range for the input of the neural network

    Parameters
    ----------
    vector_max : vector containing the lower bound for each parameter
    vector_min : vector containing the upper bound for each parameter
    num : the number of features we want to create

    Returns
    -------
    features_range : returns a random feature sample from our vector range
    """

    number_of_features = len(vector_max)

    range_linspace = np.linspace(start = vector_min, stop = vector_max, num = num)
    features_range = np.zeros((num, number_of_features))

    for i in range(number_of_features):

        np.random.shuffle(range_linspace[:, i])
        features_range[:, i] = range_linspace[:, i]

    return features_range
