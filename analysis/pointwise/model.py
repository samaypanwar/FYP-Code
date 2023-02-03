"""
Choose model then check if weights present otherwise train it and save the weights

if weights are present then test / calibrate
"""
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from analysis.gridbased import calibration_logger, logger, training_logger
from analysis.pointwise import *
from analysis.pointwise.dense import DenseModel
from helper.utils import assert_file_existence
from hyperparameters import coupon_range, maturities, maturities_label, number_of_coupon_rates, number_of_maturities, \
    test_size, train_size


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

    # Train and test sets
    params_range_train = params_range[:train_size, :]
    params_range_test = params_range[train_size: train_size + test_size, :]

    price_train = price[:train_size]
    price_test = price[train_size: train_size + test_size]

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
    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.MeanSquaredError();

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


def train_model(
        model, epochs: int = 200, batch_size: int = 30, patience: int = 20, delta: int = 0.0002,
        model_type: str = 'dense', parameterization: str = 'vasicek', plot: bool = True
        ):
    """
    This function trains our model with the given parameters and prices.
    It saves the weights of the trained model in a file at the end

    Parameters
    ----------
    model : instance of model class
    epochs : number of epochs to train our model
    batch_size : bath size for training
    patience : epochs to wait before an early stopping condition
    delta : criterion in terms of change in test loss for early stopping
    model_type : model architecture we are using ('dense')
    parameterization : parameterization that we are using for our bond pricing model
    plot : Do we want to plot our results? (True)

    Returns
    -------
    None
    """

    # Creating our train test split for data
    params_range_train, params_range_test, price_train, price_test = load_data(parameterization = parameterization)

    # Choose what type of information you want to print
    train_loss = tf.keras.metrics.Mean(name = 'train_mean')
    test_loss = tf.keras.metrics.Mean(name = 'test_mean')
    mape = tf.keras.metrics.MeanAbsolutePercentageError(name = 'mape')

    # To speed up training we need to create a some object which can send the data
    # fast to the GPU. Notice that they depend on the batch_size
    train_dataset = tf.data.Dataset.from_tensor_slices(
            (params_range_train, price_train)
            ).shuffle(10000).batch(batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices((params_range_test, price_test)).batch(batch_size)

    # TODO: ReduceLROnPlateau callback

    # Define the early stop function
    def early_stop(loss_vector):
        """
        If the cumulative product of the change in our test loss is less than our delta for the past 'patience'
        number of epochs then stop training the model
        """
        delta_loss = np.abs(np.diff(loss_vector))
        delta_loss = delta_loss[-patience:]

        return np.prod(delta_loss < delta)

    # Next we compile a few low level functions which will compute the actual gradient.

    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.MeanSquaredError()

    @tf.function
    def train_step(input_parameters, prices):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = model(input_parameters, training = True)
            loss = loss_object(prices, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)

    @tf.function
    def test_step(input_parameters, prices):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(input_parameters, training = False)
        t_loss = model.loss(prices, predictions)

        test_loss(t_loss)
        mape(prices, predictions)

    # Vectors of loss
    test_loss_vec = np.array([0.0])
    train_loss_vec = np.array([0.0])

    # We start to train the network.

    logger.info(f"Beginning training for model {model_type} with {parameterization}")
    training_logger.info(f"Beginning training for model {model_type} with {parameterization}")

    for epoch in tqdm(range(epochs)):

        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        test_loss.reset_states()
        mape.reset_states()

        for input_parameter, prices in train_dataset:
            # For each batch of images, and prices, compute the gradient and update
            # the gradient of the network
            train_step(input_parameter, prices)

        for test_images, test_prices in test_dataset:
            # For each of the test data, compute how the network performs on the data
            # We do not compute any gradients here
            test_step(test_images, test_prices)

        # Print some useful information
        template = 'Training Epoch: {0}/{1}, Loss: {2:.6f},  Test Loss: {3:.6f}, Delta Test Loss: {4:.6f}, ' \
                   'Test MAPE : {5:.3f}'
        message = template.format(
                epoch + 1,
                epochs,
                train_loss.result().numpy(),
                test_loss.result().numpy(),
                np.abs(test_loss.result() - test_loss_vec[-1]),
                mape.result().numpy()
                )

        training_logger.debug(message)

        train_loss_vec = np.append(train_loss_vec, train_loss.result())
        test_loss_vec = np.append(test_loss_vec, test_loss.result())

        if epoch > patience and early_stop(test_loss_vec):

            message = f"Early stopping at epoch = {epoch + 1}"

            training_logger.warning(message)
            logger.warning(message)

            break

    path_to_weights = f"weights/pointwise/pointwise_weights_{model_type}_{parameterization}.h5"
    assert_file_existence(path_to_weights)
    model.save_weights(path_to_weights)
    logger.info("Saved weights to file: {}".format(path_to_weights))

    plot_path = 'plotting/pointwise/'
    price_predicted_train = model(params_range_train).numpy()
    price_predicted_train = np.squeeze(price_predicted_train)
    price_predicted_test = model(params_range_test).numpy()
    price_predicted_test = np.squeeze(price_predicted_test)

    err_training_train = abs(price_predicted_train - price_train) / price_train
    err_training_test = abs(price_predicted_test - price_test) / price_test

    # Loss plots
    figure1 = plt.figure(figsize = (10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(epochs), train_loss_vec[1:], '-g')
    plt.plot(np.arange(epochs), test_loss_vec[1:], '-m')
    plt.legend(['Training Loss', 'Test Loss'])
    plt.xlabel("Epoch", fontsize = 15, labelpad = 5);
    plt.ylabel("Loss", fontsize = 15, labelpad = 5);
    text = 'Test Loss Last Epoch = %.10f' % test_loss.result().numpy() + '\n' + 'Last Epoch = %d' % (
            epochs + 1) + '\n' + 'Batch Size = %d' % batch_size
    plt.text(epochs // 4, train_loss_vec[1] / 2, text, fontsize = 12);

    figure1.savefig(f'{plot_path}pointwise_loss_{model_type}_{parameterization}.png')

    # I cluster the errors so to create the same plot as in the grid approach

    N1 = number_of_coupon_rates - 1
    N2 = number_of_maturities - 1

    mean_square_err_training_train = np.zeros((N1, N2))
    max_square_err_training_train = np.zeros((N1, N2))

    # for each maturity column
    for k in np.arange(N2):
        # get the different maturities fed as the 1st parameter in the parameter range
        # find the maturities range that lies between the maturities that we are considering
        pos_tau = (params_range_train[:, 0] >= maturities[k]) * (params_range_train[:, 0] < maturities[k + 1])

        # for each coupon rate row
        for j in np.arange(N1):
            # group by coupon rate between the given range, where coupon rate is the 2nd parameter in the parameter
            # range vector

            pos_K = (params_range_train[:, 1] >= coupon_range[j]) * (params_range_train[:, 1] < coupon_range[j + 1])

            # now that you have two ranges for our maturities and coupon rates, find the mean error for that range

            mean_square_err_training_train[j, k] = 100 * np.mean(err_training_train[pos_K * pos_tau])
            max_square_err_training_train[j, k] = 100 * np.max(err_training_train[pos_K * pos_tau])

    mean_square_err_training_test = np.zeros((N1, N2))
    max_square_err_training_test = np.zeros((N1, N2))

    # for each maturity column
    for k in np.arange(N2):
        # get the different maturities fed as the 1st parameter in the parameter range
        # find the maturities range that lies between the maturities that we are considering
        pos_tau = (params_range_test[:, 0] >= maturities[k]) * (params_range_test[:, 0] < maturities[k + 1])

        # for each coupon rate row
        for j in np.arange(N1):
            # group by coupon rate between the given range, where coupon rate is the 1st parameter in the parameter
            # range vector
            pos_K = (params_range_test[:, 1] >= coupon_range[j]) * (params_range_test[:, 1] < coupon_range[j + 1])
            # print("Num of samples in grid: ", sum(pos_K*pos_tau))
            # now that you have two ranges for our maturities and coupon rates, find the mean error for that range

            try:
                mean_square_err_training_test[j, k] = 100 * np.mean(err_training_test[pos_K * pos_tau])
                max_square_err_training_test[j, k] = 100 * np.max(err_training_test[pos_K * pos_tau])

            except:
                mean_square_err_training_test[j, k] = 0
                max_square_err_training_test[j, k] = 0

    # Heatmap train loss

    mean_square_err_training_train = pd.DataFrame(mean_square_err_training_train).T
    max_square_err_training_train = pd.DataFrame(max_square_err_training_train).T

    mean_square_err_training_train['Maturity'] = maturities_label
    max_square_err_training_train['Maturity'] = maturities_label

    mean_square_err_training_train.set_index('Maturity', inplace = True)
    max_square_err_training_train.set_index('Maturity', inplace = True)

    mean_square_err_training_train.columns = coupon_range[1:]
    max_square_err_training_train.columns = coupon_range[1:]

    figure_train, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (16, 7))
    figure_train.suptitle("Training Error", fontsize = 15)
    sns.heatmap(
            mean_square_err_training_train, annot = True, fmt = ".2f", linewidths = 0.0, cmap = 'viridis',
            ax = ax1
            )

    ax1.set_xlabel("Coupon Rate")
    sns.heatmap(
            max_square_err_training_train, annot = True, fmt = ".2f", linewidths = 0.0, cmap = 'viridis', ax = ax2,
            )
    ax2.set_xlabel("Coupon Rate")
    ax1.title.set_text('Average Percentage Error')
    ax2.title.set_text('Maximum Percentage Error')
    plt.tight_layout()
    plt.show()

    figure_train.savefig(
            f'{plot_path}pointwise_error_train_{model_type}_{parameterization}.png', bbox_inches = 'tight',
            pad_inches = 0.01
            )

    # Heatmap test loss
    mean_square_err_training_test = pd.DataFrame(mean_square_err_training_test).T
    max_square_err_training_test = pd.DataFrame(max_square_err_training_test).T

    mean_square_err_training_test['Maturity'] = maturities_label
    max_square_err_training_test['Maturity'] = maturities_label

    mean_square_err_training_test.set_index('Maturity', inplace = True)
    max_square_err_training_test.set_index('Maturity', inplace = True)

    mean_square_err_training_test.columns = coupon_range[1:]
    max_square_err_training_test.columns = coupon_range[1:]

    figure_test, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (16, 7))
    figure_test.suptitle("Test Error", fontsize = 15)
    sns.heatmap(
            mean_square_err_training_test, annot = True, fmt = ".2f", linewidths = 0.0, cmap = 'viridis',
            ax = ax1
            )

    ax1.set_xlabel("Coupon Rate")
    sns.heatmap(
            max_square_err_training_test, annot = True, fmt = ".2f", linewidths = 0.0, cmap = 'viridis', ax = ax2,
            )
    ax2.set_xlabel("Coupon Rate")
    ax1.title.set_text('Average Percentage Error')
    ax2.title.set_text('Maximum Percentage Error')
    plt.tight_layout()
    plt.show()
    figure_test.savefig(
            f'{plot_path}pointwise_error_test_{model_type}_{parameterization}.png', bbox_inches =
            'tight', pad_inches = 0.01
            )


def calibrate_synthetic(
        model, calibration_size: int = 10_000, epochs: int = 1000, model_type: str = 'dense', parameterization:
        str =
        'vasicek',
        plot: bool = False,
        verbose_length: int = 1
        ):
    """

    Parameters
    ----------
    model : instance of model class
    calibration_size : size of the calibration set
    epochs: number of epochs to calibrate
    model_type : model architecture we are using (either 'dense' or 'cnn')
    parameterization : parameterization that we are using for our forward curve model
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

    parameters_to_calibrate = np.loadtxt(f'data/pointwise/pointwise_parameters_{parameterization}.dat')
    prices_calibrate = np.reshape(
            np.loadtxt(f'data/pointwise/pointwise_price_{parameterization}.dat'),
            newshape = (train_size + test_size, 1), order = 'F'
            )

    parameters_to_calibrate = parameters_to_calibrate[train_size + np.arange(calibration_size), :]
    prices_calibrate = prices_calibrate[train_size + np.arange(calibration_size)]

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

    # TODO: don't induce errors in the fixed columns
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
    prices = tf.convert_to_tensor(prices_calibrate)

    logger.info(f"Beginning calibration for model {model_type} with {parameterization}")
    calibration_logger.info(f"Beginning calibration for model {model_type} with {parameterization}")

    ta = tf.TensorArray(tf.float64, size = 0, dynamic_size = True, clear_after_read = False)

    # Start the actual calibration
    for j in tqdm(range(calibration_size)):

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

    np.savetxt(f'data/pointwise/pointwise_params_calibrated_{model_type}_{parameterization}.dat', ta.stack().numpy())
    logger.info(
            f"Saved parameters to file: "
            f"{f'data/pointwise/pointwise_params_calibrated_{model_type}_{parameterization}.dat'}"
            )
    #
    # Errors and plots
    new_input_guess = ta.stack().numpy().copy()
    percentage_err = np.abs(new_input_guess - parameters_to_calibrate) / np.abs(parameters_to_calibrate)
    mean_percentage_err = np.mean(percentage_err, axis = 0) * 100
    percentage_err_copy = percentage_err.copy()
    percentage_err_copy.sort(axis = 0)
    median_percentage_err = percentage_err_copy[calibration_size // 2, :] * 100

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
                2 + i] + r'%'
            plt.text(
                    np.mean(parameters_to_calibrate[:, 2 + i]), np.max(percentage_err[:, 2 + i]*90), s2, fontsize = 15,
                    weight = 'bold'
                    )

        f.savefig(
                f'plotting/pointwise/pointwise_calibrated_{model_type}_{parameterization}.png', bbox_inches = 'tight',
                pad_inches =
                0.01
                )

def calibrate_to_market_data(
        model, market_data, initial_parameters, time_to_expiry, epochs: int = 1000, model_type: str = 'dense',
        parameterization:
        str =
        'vasicek',
        verbose_length: int = 1
        ):

    # The network is done training. We are ready to start on the Calibration step
    if parameterization == 'vasicek':
        parameter_size = 4

    else:
        logger.error("Unknown parameterization: %s" % parameterization)
        raise ValueError("Unknown parameterization")

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

    logger.info(f"Beginning calibration for model {model_type} with {parameterization}")
    calibration_logger.info(f"Beginning calibration for model {model_type} with {parameterization}")

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

        # print(time_to_expiry[j])
        # print(parameters[0])
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

    np.savetxt(f'data/pointwise/pointwise_params_market_calibrated_{model_type}_{parameterization}.dat',
               ta.stack().numpy())
    logger.info(
            f"Saved parameters to file: "
            f"{f'data/pointwise/pointwise_params_calibrated_{model_type}_{parameterization}.dat'}"
            )

