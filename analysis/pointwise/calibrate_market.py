"""
This file contains the calibration of our model where we try to recover the original underlying model parameters
given the observed prices
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

path_parent = os.path.dirname(os.getcwd())

if os.getcwd()[-8:] != 'FYP-Code':
    os.chdir(path_parent)

path = os.getcwd()
sys.path.append(path)

from analysis.pointwise import calibration_logger, logger
from analysis.pointwise.utils import init_model, load_weights
from hyperparameters import coupons, maturities_label, optimizer, loss_object

plt.style.use('seaborn-v0_8')
plt.rcParams.update({'font.family': 'Times New Roman'})
plt.rcParams.update({'axes.grid': True, 'axes.linewidth': 0.5, 'axes.edgecolor': 'black'})


def calibrate_to_market_data(
        model, market_data, initial_parameters, time_to_expiry, epochs: int = 1000, model_type: str = 'dense',
        parameterization:
        str =
        'two_factor',
        maturity: str = '1Y',
        verbose_length: int = 1,
        ):

    parameters_to_calibrate = initial_parameters
    prices_calibrate = market_data

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

    # Important: First convert to tensor, then to variable
    prices = tf.convert_to_tensor(prices_calibrate)
    calibration_size = len(prices_calibrate)

    logger.info(f"Beginning calibration for model {model_type} with {parameterization} for {maturity} maturity")
    calibration_logger.info(
            f"Beginning calibration for model {model_type} with {parameterization} for {maturity} maturity"
            )

    ta = tf.TensorArray(tf.float64, size = 0, dynamic_size = True, clear_after_read = False)

    parameters = initial_parameters.copy()[1:]

    # Start the actual calibration
    for j in tqdm(range(calibration_size), desc = f'Calibrating {maturity}...'):

        variable_input_guess = tf.Variable(
                tf.reshape(
                        tf.convert_to_tensor(parameters), shape = (1,
                                                                   -1)
                        )
                )

        fixed_input_guess = np.array([time_to_expiry[j], initial_parameters[0]], dtype = np.float64)

        fixed_input_guess = tf.reshape(
                tf.convert_to_tensor(fixed_input_guess), shape = (1,
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

        parameters = ta.read(j).numpy()[2:]

    change = ta.read(j).numpy()[2:] - initial_parameters[1:]

    message = f"Calibration complete! change in parameters: {np.linalg.norm(change)}"
    logger.info(message)
    calibration_logger.info(message)

    np.savetxt(
            f'data/pointwise/market_calibrated_parameters_{maturity}.dat',
            ta.stack().numpy()
            )
    logger.info(
            f"Saved parameters to file: "
            f"{ f'data/pointwise/market_calibrated_parameters_{maturity}.dat'}"
            )

    calib = pd.read_table(
             f'data/pointwise/market_calibrated_parameters_{maturity}.dat', sep = " ", header = None
            )
    df = pd.read_csv(f'market_data/{maturity}_cleaned.csv')
    fig, ax = plt.subplots(nrows = 1, ncols = 1)

    (100 * (1/calib.loc[:, 4] + (1/calib.loc[:, 5]))).plot(ax = ax, label = 'Predicted Rate');
    ax.plot(100*df.Price, label = 'Market Rate');
    ax.legend();
    ax.set_title(f'Maturity: {maturity}')

    fig.savefig(
            f'plotting/pointwise/market_calibrated_{maturity}_rates.png', bbox_inches = 'tight',
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
            '-m', "--maturity", type = str,
            help = "Maturity of bond contract", default = 'NA', choices = maturities_label
            )

    args = parser.parse_args()

    model = init_model(parameterization = args.parameterisation)
    load_weights(model)

    if args.maturity != 'NA':

        df = pd.read_csv(f'market_data/{args.maturity}_cleaned.csv')

        c = coupons[args.maturity]
        b = 5
        a = (b * df.Price[0]) / 100
        sigma = 0.3
        initial_parameters = [c, a, b, sigma]

        calibrate_to_market_data(
                model = model,
                market_data = df['Bond Price'],
                time_to_expiry = df['Time to Expiry'],
                initial_parameters = np.array(initial_parameters, dtype = np.float64),
                epochs = args.epochs,
                model_type = 'dense',
                parameterization = args.parameterisation,
                verbose_length = args.verbose_length,
                maturity = args.maturity,
                )

    else:
        for maturity in maturities_label:

            df = pd.read_csv(f'market_data/{maturity}_cleaned.csv')

            c = coupons[maturity]
            b = 100
            a = 100
            sigma = 0.3
            rho = 0
            eta = 0.3
            x = 0.005
            y = 0.005

            initial_parameters = [c, x, y, a, b, sigma, eta, rho]

            calibrate_to_market_data(
                    model = model,
                    market_data = df['Bond Price'],
                    time_to_expiry = df['Time to Expiry'],
                    initial_parameters = np.array(initial_parameters, dtype = np.float64),
                    epochs = args.epochs,
                    model_type = 'dense',
                    parameterization = args.parameterisation,
                    verbose_length = args.verbose_length,
                    maturity = maturity,
                    )
