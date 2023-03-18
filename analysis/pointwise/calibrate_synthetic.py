"""
This file contains the calibration of our model where we try to recover the original underlying model parameters
given the observed prices
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import tensorflow as tf
from tqdm import tqdm

path_parent = os.path.dirname(os.getcwd())

if os.getcwd()[-8:] != 'FYP-Code':
    os.chdir(path_parent)

path = os.getcwd()
sys.path.append(path)

from analysis.pointwise import calibration_logger, logger
from analysis.pointwise.utils import load_weights, init_model
from hyperparameters import test_size, train_size, optimizer, loss_object

plt.style.use('seaborn-v0_8')
plt.rcParams.update({'font.family': 'Times New Roman'})
plt.rcParams.update({'axes.grid': True, 'axes.linewidth': 0.5, 'axes.edgecolor': 'black'})

def calibrate_synthetic(
        model, calibration_size: int = 10_000, epochs: int = 1000, model_type: str = 'dense', parameterization:
        str =
        'two_factor',
        verbose_length: int = 1
        ):
    """

    Parameters
    ----------
    model : instance of model class
    calibration_size : size of the calibration set
    epochs: number of epochs to calibrate
    model_type : model architecture we are using ('dense')
    parameterization : parameterization that we are using for our bond pricing model ('vasicek')
    plot : Do we want to plot our results? (True)

    Returns
    -------
    None

    """
    # The network is done training. We are ready to start on the Calibration step
    if parameterization == 'two_factor':
        parameter_size = 8

    else:
        logger.error("Unknown parameterization: %s" % parameterization)
        raise ValueError("Unknown parameterization")

    # Loading in the prices/parameters from our testing set
    parameters_to_calibrate = np.loadtxt(f'data/pointwise/pointwise_parameters_{parameterization}.dat')
    prices_calibrate = np.loadtxt(f'data/pointwise/pointwise_price_{parameterization}.dat')

    # finding the number of samples
    samples = prices_calibrate.shape[0]

    train_proportion = round(train_size / (train_size + test_size), 2)

    number_of_training_samples = round(samples * train_proportion)

    parameters_to_calibrate = parameters_to_calibrate[number_of_training_samples + np.arange(calibration_size), :]
    prices_calibrate = prices_calibrate[number_of_training_samples + np.arange(calibration_size)]

    # Choose optimizer and type of loss function
    calibration_loss = tf.keras.metrics.Mean(name = 'calibration_mean')
    mape = tf.keras.metrics.MeanAbsolutePercentageError(name = 'mape')

    @tf.function
    def calibration_step(fixed_input, variable_input, price):

        with tf.GradientTape() as tape:
            tape.watch(variable_input)

            prediction = model(tf.concat([fixed_input, variable_input], axis = 1))
            c_loss = loss_object(price, prediction)
            calibration_loss(c_loss)
            grads = tape.gradient(c_loss, [variable_input])
            optimizer.apply_gradients(zip(grads, [variable_input]))

        mape(price, prediction)

    # We need to guess some initial model parameters. We induce errors in our old guesses here as a test
    parameters_with_errors = parameters_to_calibrate + parameters_to_calibrate * np.concatenate(
            (
                    np.zeros(shape = (calibration_size, 2)),
                    np.random.rand(
                            calibration_size,
                            parameter_size
                            ) * np.array([0.05] * parameter_size)),
            axis = 1
            )
    # I just copy the starting parameters for convenience. This is not necessary
    old_input_guess = parameters_with_errors.copy()

    # Important: First convert to tensor, then to variable
    prices = tf.convert_to_tensor(prices_calibrate)

    logger.info(f"Beginning calibration for model {model_type} with {parameterization}")
    calibration_logger.info(f"Beginning calibration for model {model_type} with {parameterization}")

    ta = tf.TensorArray(tf.float64, size = 0, dynamic_size = True, clear_after_read = False)

    # Start the actual calibration
    for j in tqdm(range(calibration_size), desc = 'Calibrating...'):

        variable_input_guess = tf.Variable(
                tf.reshape(
                        tf.convert_to_tensor(parameters_with_errors[j, 2:]), shape = (1,
                                                                                      -1)
                        )
                )
        fixed_input_guess = tf.reshape(
                tf.convert_to_tensor(parameters_with_errors[j, :2]), shape = (1,
                                                                              -1)
                )

        for epoch in range(epochs):
            calibration_loss.reset_states()
            mape.reset_states()

            calibration_step(fixed_input_guess, variable_input_guess, prices[j])

            template = 'Set: {0}/{1} Calibration Epoch: {2}/{3}, Loss: {4}, MAPE: {5}'
            message = template.format(
                    j, calibration_size, epoch + 1, epochs, calibration_loss.result(), mape.result()
                    )

            # So we are not logging very frequently
            if (epoch + 1) % verbose_length == 0 and epoch > 1:
                calibration_logger.debug(message)

        ta.write(
                j, tf.reshape(
                        tf.concat([fixed_input_guess, variable_input_guess], axis = 1), shape = (-1,
                                                                                                 )
                        )
                ).mark_used()

    change = ta.stack().numpy() - old_input_guess

    message = f"Calibration complete! change in parameters: {np.linalg.norm(change, 'fro')}"
    logger.info(message)
    calibration_logger.info(message)

    np.savetxt(f'data/pointwise/calibrated_parameters.dat', ta.stack().numpy())
    logger.info(
            f"Saved parameters to file: "
            f"{f'data/pointwise/calibrated_parameters.dat'}"
            )
    #
    # Errors and plots
    new_input_guess = ta.stack().numpy().copy()
    percentage_err = np.abs(new_input_guess - parameters_to_calibrate) / np.abs(parameters_to_calibrate)
    mean_percentage_err = np.mean(percentage_err, axis = 0) * 100
    percentage_err_copy = percentage_err.copy()
    percentage_err_copy.sort(axis = 0)
    median_percentage_err = percentage_err_copy[calibration_size // 2, :] * 100

    absolute_error = np.abs(new_input_guess - parameters_to_calibrate)
    mae = np.mean(absolute_error, axis = 0)

    f = plt.figure(figsize = (20, 15))
    parameter_names = ['phi', 'x', 'y', 'a', "b", "sigma", 'eta', 'rho']

    for i in range(parameter_size):

        plt.subplot(3, 3, 1 + i)
        plt.plot(parameters_to_calibrate[:, 2 + i], percentage_err[:, 2 + i] * 100, '*', color = 'midnightblue')
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
        plt.title(f'${parameter_names[i]}$')
        plt.ylabel('Percentage error')
        s2 = 'Average: %.2f' % mean_percentage_err[2 + i] + r'%' + '\n' + 'Median: %.2f' % median_percentage_err[
            2 + i] + r'%' + '\n' + 'MAE: %.2f' % mae[2 + i]

        plt.text(0.73,
                 0.95,
                s2,
                fontsize = 15,
                weight = 'bold', ha = 'left', va = 'top', transform=plt.gca().transAxes
                )

        plt.tight_layout()

    f.savefig(
            f'plotting/pointwise/calibrated_scatterplot.png', bbox_inches = 'tight',
            pad_inches =
            0.3
            )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
            '-p', "--parameterisation", type = str,
            help = "Parameterisation for our underlying bond pricing model", default = 'two_factor',
            choices = ['two_factor']
            )

    parser.add_argument(
            '-e', "--epochs", type = int,
            help = "Epochs for training model", default = 200,
            )

    parser.add_argument(
            '-v', "--verbose-length", type = int,
            help = "Frequency of logging for each set", default = 100,
            )

    parser.add_argument(
            '-c', "--calibration-size", type = int,
            help = "Size of the calibration set", default = 1000,
            )

    args = parser.parse_args()

    model = init_model(parameterization = args.parameterisation)
    load_weights(model)

    calibrate_synthetic(
            model = model,
            epochs = args.epochs,
            model_type = 'dense',
            parameterization = args.parameterisation,
            calibration_size = args.calibration_size,
            verbose_length = args.verbose_length
            )
