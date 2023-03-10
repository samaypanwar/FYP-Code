import os
import numpy as np

from analysis.pointwise import logger
import tensorflow as tf
from analysis.pointwise.dense import DenseModel

from hyperparameters import test_size, train_size, optimizer, loss_object

def load_data(parameterization: str = 'vasicek'):
    """
    This function loads up our data for the prices/parameters and creates train test splits on the same

    Parameters
    ----------
    parameterization : parameterization that we are using for our bond pricing model

    Returns
    -------
    train test splits for prices/parameters
    """

    params_range = np.loadtxt(f'data/pointwise/pointwise_parameters_{parameterization}.dat')
    price = np.loadtxt(f'data/pointwise/pointwise_price_{parameterization}.dat')

    samples = params_range.shape[0]

    train_proportion = round(train_size / (train_size + test_size), 2)

    # Train and test sets
    params_range_train = params_range[:round(samples * train_proportion), :]
    params_range_test = params_range[round(samples * train_proportion):, :]

    price_train = price[:round(samples * train_proportion)]
    price_test = price[round(samples * train_proportion):]

    return params_range_train, params_range_test, price_train, price_test


def init_model(model_type: str = 'dense', parameterization: str = 'vasicek'):
    """
    This function initializes and compiles our model type with the stated optimizer and loss function

    Parameters
    ----------
    model_type : model architecture we are using ('dense')
    parameterization: parameterization that we are using for our bond pricing model

    Returns
    -------
    Instance of built model class

    """

    # Create an instance of the model
    if model_type.lower() == 'dense':
        model = DenseModel()

    else:
        logger.error("Unknown model type: %s" % model_type)
        raise ValueError("Unknown model type")

    if parameterization == 'vasicek':
        """
        The number of parameters we consider in our model in pointwise are:
        a: vasicek process parameter
        b: vasicek process parameter
        sigma: vasicek process parameter
        r: yield of our bond

        It returns the price of the bond
        """
        parameter_size = 4

    else:
        logger.error("Unknown parameterization: %s" % parameterization)
        raise ValueError("Unknown parameterization")

    # Choose optimizer and type of loss function

    model.compile(
            loss = loss_object,
            optimizer = optimizer
            )

    #  tau: time to maturity for our bond + coupon rate, which is not to be calibrated
    model.build(input_shape = (1, parameter_size + 2))
    model.summary()

    logger.info(f"Model of type: {model_type} has been initialized")

    return model


def load_weights(model, model_type: str = 'dense', parameterization: str = 'vasicek'):
    """
    This function loads the weight of a trained model into the compiled model if they exist

    Parameters
    ----------
    model : instance of model class
    model_type : model architecture we are using ('dense')
    parameterization : parameterization that we are using for our bond pricing model

    Returns
    -------
    None
    """

    # Path to our weights file
    path = f"weights/pointwise/pointwise_weights_{model_type}_{parameterization}.h5"

    # Check if the weights file exists
    if os.path.isfile(path):
        model.load_weights(path)
        logger.info("Weights loaded successfully")

    else:
        logger.error("Failed to load the weights file: {}".format(path))
        raise FileNotFoundError("Please train the weights. No such file or directory {0}".format(path))


