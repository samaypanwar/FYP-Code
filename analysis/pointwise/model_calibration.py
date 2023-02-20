"""
This file contains the calibration of our model where we try to recover the original underlying model parameters
given the observed prices
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from analysis.pointwise import calibration_logger, logger
from hyperparameters import test_size, train_size


def calibrate_synthetic(
        model, calibration_size: int = 10_000, epochs: int = 1000, model_type: str = 'dense', parameterization:
        str =
        'vasicek',
        plot: bool = True,
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
    if parameterization == 'vasicek':
        parameter_size = 4

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

    # We need to guess some initial model parameters. We induce errors in our old guesses here as a test
    parameters_with_errors = parameters_to_calibrate + np.concatenate(
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

    logger.info(f"Beginning calibration for model {model_type} with {parameterization}")
    calibration_logger.info(f"Beginning calibration for model {model_type} with {parameterization}")

    from scipy.optimize import Bounds

    variable_bounds = {
            'a'    : [0.01, 0.20],
            'b'    : [1, 10],
            'sigma': [0.1, 1],
            'r'    : [0.00, 0.1]
            }

    bounds = Bounds(
            [variable_bounds[key][0] for key in variable_bounds], [variable_bounds[key][1] for key in variable_bounds]
            )

    from sklearn.metrics import mean_squared_error as mse

    def objective_function(variable_input, fixed_input, model, y_true):

        y_pred = model(
                np.reshape(
                        np.concatenate([fixed_input, variable_input])
                        , newshape = (1, -1)
                        )
                )[0].numpy()

        return mse(y_true, y_pred)

    import scipy

    updated_variable_input = np.empty(shape = (1, 4))

    for i in tqdm(range(calibration_size)):

        variable_input = parameters_with_errors[i, 2:]
        fixed_input = parameters_with_errors[i, :2]
        y_true = np.array([prices_calibrate[i]])
        args = (fixed_input, model, y_true)
        x0 = variable_input
        s = scipy.optimize.minimize(
                objective_function, x0, method = 'trust-constr',
                options = {'verbose': 0}, bounds = bounds, args = args
                )

        updated_variable_input = np.append(updated_variable_input, s.x.reshape((1, -1)), axis = 0)
        text = f"Set {i+1}/{calibration_size}, Time Elapsed: {round(s.execution_time, 3)} seconds"
        calibration_logger.debug(text)

    updated_variable_input = np.concatenate([parameters_with_errors[:, :2], updated_variable_input[1:, :]], axis=1)

    change = updated_variable_input - old_input_guess

    message = f"Calibration complete! change in parameters: {np.linalg.norm(change, 'fro')}"
    logger.info(message)
    calibration_logger.info(message)

    np.savetxt(f'data/pointwise/pointwise_params_calibrated_{model_type}_{parameterization}.dat', updated_variable_input)
    logger.info(
            f"Saved parameters to file: "
            f"{f'data/pointwise/pointwise_params_calibrated_{model_type}_{parameterization}.dat'}"
            )
    #
    # Errors and plots
    new_input_guess = updated_variable_input
    percentage_err = np.abs(new_input_guess - parameters_to_calibrate) / np.abs(parameters_to_calibrate)
    mean_percentage_err = np.mean(percentage_err, axis = 0) * 100
    percentage_err_copy = percentage_err.copy()
    percentage_err_copy.sort(axis = 0)
    median_percentage_err = percentage_err_copy[calibration_size // 2, :] * 100

    absolute_error = np.abs(new_input_guess - parameters_to_calibrate)
    mae = np.mean(absolute_error, axis = 0)

    if plot:

        f = plt.figure(figsize = (20, 15))
        parameter_names = ['a', "b", "sigma", "r"]

        for i in range(parameter_size):

            plt.subplot(2, 2, 1 + i)
            plt.plot(parameters_to_calibrate[:, 2 + i], percentage_err[:, 2 + i] * 100, '*', color = 'midnightblue')
            plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
            plt.title(f'${parameter_names[i]}$')
            plt.ylabel('Percentage error')
            s2 = 'Average: %.2f' % mean_percentage_err[2 + i] + r'%' + '\n' + 'Median: %.2f' % median_percentage_err[
                2 + i] + r'%' + '\n' + 'MAE: %.2f' % mae[2 + i]

            plt.text(
                    np.mean(parameters_to_calibrate[:, 2 + i]), np.max(percentage_err[:, 2 + i] * 90), s2,
                    fontsize = 15,
                    weight = 'bold'
                    )

        f.savefig(
                f'plotting/pointwise/pointwise_calibrated_{model_type}_{parameterization}.png', bbox_inches = 'tight',
                pad_inches =
                0.3
                )


def calibrate_to_market_data(
        model, market_data, initial_parameters, time_to_expiry, epochs: int = 1000, model_type: str = 'dense',
        parameterization:
        str =
        'vasicek',
        maturity: str = '1M',
        verbose_length: int = 1
        ):

    parameters_to_calibrate = initial_parameters
    prices_calibrate = market_data

    # Choose optimizer and type of loss function
    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.MeanSquaredError()
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

    parameters = initial_parameters.copy()

    # Start the actual calibration
    for j in tqdm(range(calibration_size)):

        variable_input_guess = tf.Variable(
                tf.reshape(
                        tf.convert_to_tensor(parameters[1:]), shape = (1,
                                                                       -1)
                        )
                )

        fixed_input_guess = np.array([time_to_expiry[j], parameters[0]], dtype = np.float64)
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

        parameters = ta.read(j).numpy()[1:]

    change = ta.read(j).numpy()[1:] - initial_parameters

    message = f"Calibration complete! change in parameters: {np.linalg.norm(change)}"
    logger.info(message)
    calibration_logger.info(message)

    np.savetxt(
        f'data/pointwise/pointwise_params_market_calibrated_{model_type}_{parameterization}_{maturity}.dat',
        ta.stack().numpy()
        )
    logger.info(
            f"Saved parameters to file: "
            f"{f'data/pointwise/pointwise_params_calibrated_{model_type}_{parameterization}_{maturity}.dat'}"
            )
