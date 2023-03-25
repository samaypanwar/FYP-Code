"""
Choose model then check if weights present otherwise train it and save the weights

if weights are present then test / calibrate
"""
import os
from itertools import product

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from tqdm import tqdm

from analysis.pointwise import *
from analysis.pointwise.convolutional import CNN
from analysis.pointwise.dense import DenseModel

plt.style.use('seaborn-v0_8-pastel')
plt.rcParams.update({'font.family': 'Times New Roman'})
plt.rcParams.update({'axes.grid': True, 'axes.linewidth': 0.5, 'axes.edgecolor': 'black'})



def load_data(parameterization: str = 'nelson_siegel'):
    """
    This function loads up our data for the prices/parameters and creates train test splits on the same

    Parameters
    ----------
    parameterization : parameterization that we are using for our forward curve model

    Returns
    -------
    train test splits for prices/parameters
    """

    params_range = np.loadtxt(f'data/pointwise/pointwise_parameters_{parameterization}.dat')
    price = np.loadtxt(f'data/pointwise/pointwise_price_{parameterization}.dat')

    # Train and test sets
    params_range_train = params_range[:train_size, :]  # size=[train_size, param_in]
    params_range_test = params_range[train_size: train_size + test_size, :]  # size=[test_size, param_in]

    price_train = price[:train_size]  # size=[train_size, N1, N2]
    price_test = price[train_size: train_size + test_size]  # size=[test_size, N1, N2]

    return params_range_train, params_range_test, price_train, price_test


def init_model(model_type: str = 'dense', parameterization: str = 'nelson_siegel'):
    """
    This function initializes and compiles our model type with the stated optimizer and loss function

    Parameters
    ----------
    model_type : model architecture we are using (either 'dense' or 'cnn')
    parameterization: parameterization that we are using for our forward curve model

    Returns
    -------
    Instance of built model class
    """

    # Create an instance of the model
    if model_type.lower() == 'dense':
        model = DenseModel()

    elif model_type.lower() == 'cnn':
        model = CNN()

    else:
        logger.error("Unknown model type: %s" % model_type)
        raise ValueError("Unknown model type")

    if parameterization == 'nelson_siegel':
        parameter_size = 7

    elif parameterization == 'svensson':
        parameter_size = 9


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
    model.build(input_shape = (1, parameter_size + 2))
    model.summary()

    logger.info(f"Model of type: {model_type} has been initialized")

    return model


def load_weights(model, model_type: str = 'dense', parameterization: str = 'nelson_siegel'):
    """
    This function loads the weight of a trained model into the compiled model if they exist

    Parameters
    ----------
    model : instance of model class
    model_type : model architecture we are using (either 'dense' or 'cnn')
    parameterization : parameterization that we are using for our forward curve model

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
        model_type: str = 'dense', parameterization: str = 'nelson_siegel', plot: bool = True
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
    model_type : model architecture we are using (either 'dense' or 'cnn')
    parameterization : parameterization that we are using for our forward curve model
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

    # To speed up training we need to create a some object which can send the data
    # fast to the GPU. Notice that they depend on the batch_size
    train_dataset = tf.data.Dataset.from_tensor_slices(
            (params_range_train, price_train)
            ).shuffle(10000).batch(batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices((params_range_test, price_test)).batch(batch_size)

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

        for input_parameter, prices in train_dataset:
            # For each batch of images, and prices, compute the gradient and update
            # the gradient of the network
            train_step(input_parameter, prices)

        for test_images, test_prices in test_dataset:
            # For each of the test data, compute how the network performs on the data
            # We do not compute any gradients here
            test_step(test_images, test_prices)

        # Print some useful information
        template = 'Training Epoch {0}/{1}, Loss: {2:.6f},  Test Loss: {3:.6f}, Delta Test Loss: {4:.6f}'
        message = template.format(
                epoch + 1,
                epochs,
                train_loss.result().numpy(),
                test_loss.result().numpy(),
                np.abs(test_loss.result() - test_loss_vec[-1])
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
    model.save_weights(path_to_weights)
    logger.info("Saved weights to file: {}".format(path_to_weights))

    if plot:
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
        # TODO: Init of time to contract start and strikes
        K_vector = np.array([31.6, 31.7, 31.9, 32.1, 32.3, 32.5, 32.7, 32.9, 33.1, 33.2])
        tau_vector = np.array(
                [1 / 12, 1 / 12 + 1 / 24, 2 / 12 + 1 / 24, 3 / 12 + 1 / 24, 4 / 12 + 1 / 24, 5 / 12 + 1 / 24,
                 6 / 12 + 1 / 24, 1]
                )
        N1 = len(K_vector) - 1
        N2 = len(tau_vector) - 1

        mean_square_err_training_train = np.zeros((N1, N2))
        max_square_err_training_train = np.zeros((N1, N2))
        for k in np.arange(N2):
            pos_tau = (params_range_train[:, 1] >= tau_vector[k]) * (params_range_train[:, 1] < tau_vector[k + 1])
            for j in np.arange(N1):
                pos_K = (params_range_train[:, 0] >= K_vector[j]) * (params_range_train[:, 0] < K_vector[j + 1])
                mean_square_err_training_train[j, k] = 100 * np.mean(err_training_train[pos_K * pos_tau])
                max_square_err_training_train[j, k] = 100 * np.max(err_training_train[pos_K * pos_tau])

        mean_square_err_training_test = np.zeros((N1, N2))
        max_square_err_training_test = np.zeros((N1, N2))
        for k in np.arange(N2):
            pos_tau = (params_range_test[:, 1] >= tau_vector[k]) * (params_range_test[:, 1] < tau_vector[k + 1])
            for j in np.arange(N1):
                pos_K = (params_range_test[:, 0] >= K_vector[j]) * (params_range_test[:, 0] < K_vector[j + 1])
                mean_square_err_training_test[j, k] = 100 * np.mean(err_training_test[pos_K * pos_tau])
                max_square_err_training_test[j, k] = 100 * np.max(err_training_test[pos_K * pos_tau])

        # Heatmap train loss
        K_label = np.array([31.6, 31.8, 32.0, 32.2, 32.4, 32.6, 32.8, 33.0, 33.2])
        tau_label = ['1', '2', '3', '4', '5', '6', '12']

        figure_train, ax = plt.subplots(figsize = (15, 5), ncols = 2)

        ax[0].set_title("Average percentage error", fontsize = 15, y = 1.04)
        im = ax[0].imshow(np.transpose(mean_square_err_training_train))
        figure_train.colorbar(im, ax=ax[0], format = mtick.PercentFormatter(), pad = 0.01, fraction = 0.046,
                              )
        ax[0].set_xticks(np.linspace(0, N1 - 1, N1))
        ax[0].set_xticklabels(K_label)
        ax[0].set_yticks(np.linspace(0, N2 - 1, N2))
        ax[0].set_yticklabels(tau_label)
        ax[0].set_xlabel("Strike", fontsize = 15, labelpad = 5);
        ax[0].set_ylabel("Maturity (month)", fontsize = 15, labelpad = 5);

        ax[1].set_title("Maximum percentage error", fontsize = 15, y = 1.04)
        im2 = ax[1].imshow(np.transpose(max_square_err_training_train))
        figure_train.colorbar(im2, ax=ax[1], format = mtick.PercentFormatter(), pad = 0.01, fraction = 0.046)
        ax[1].set_xticks(np.linspace(0, N1 - 1, N1))
        ax[1].set_xticklabels(K_label)
        ax[1].set_yticks(np.linspace(0, N2 - 1, N2))
        ax[1].set_yticklabels(tau_label)
        ax[1].set_xlabel("Strike", fontsize = 15, labelpad = 5);
        ax[1].set_ylabel("Maturity (month)", fontsize = 15, labelpad = 5);

        figure_train.suptitle('Training Error', fontsize = 15)

        figure_train.savefig(
                f'{plot_path}pointwise_error_train_{model_type}_{parameterization}.png', bbox_inches = 'tight',
                pad_inches = 0.01
                )

        # Heatmap test loss
        figure_test, ax = plt.subplots(ncols = 2, figsize = (15, 5))
        ax[0].set_title("Average percentage error", fontsize = 15, y = 1.04)
        im = ax[0].imshow(np.transpose(mean_square_err_training_test))
        figure_test.colorbar(im, ax=ax[0], format = mtick.PercentFormatter(), pad = 0.01, fraction = 0.046)
        ax[0].set_xticks(np.linspace(0, N1 - 1, N1))
        ax[0].set_xticklabels(K_label)
        ax[0].set_yticks(np.linspace(0, N2 - 1, N2))
        ax[0].set_yticklabels(tau_label)
        ax[0].set_xlabel("Strike", fontsize = 15, labelpad = 5);
        ax[0].set_ylabel("Maturity (month)", fontsize = 15, labelpad = 5);

        ax[1].set_title("Maximum percentage error", fontsize = 15, y = 1.04)
        im2 = ax[1].imshow(np.transpose(max_square_err_training_test))
        figure_test.colorbar(im2, ax=ax[1], format = mtick.PercentFormatter(), pad = 0.01, fraction = 0.046)
        ax[1].set_xticks(np.linspace(0, N1 - 1, N1))
        ax[1].set_xticklabels(K_label)
        ax[1].set_yticks(np.linspace(0, N2 - 1, N2))
        ax[1].set_yticklabels(tau_label)
        ax[1].set_xlabel("Strike", fontsize = 15, labelpad = 5);
        ax[1].set_ylabel("Maturity (month)", fontsize = 15, labelpad = 5);

        figure_test.suptitle('Test Error', fontsize = 15)

        figure_test.savefig(
                f'{plot_path}pointwise_error_test_{model_type}_{parameterization}.png', bbox_inches =
                'tight', pad_inches = 0.01
                )


def calibrate(
        model, prices, parameters, epochs: int = 1000, model_type: str = 'dense', parameterization: str =
        'nelson_siegel',
        plot: bool = False
        ):
    """

    Parameters
    ----------
    model : instance of model class
    prices : prices to calibrate our parameters to
    parameters : old set of parameters to act as initial estimates
    epochs: number of epochs to calibrate
    model_type : model architecture we are using (either 'dense' or 'cnn')
    parameterization : parameterization that we are using for our forward curve model
    plot : Do we want to plot our results? (True)

    Returns
    -------
    None

    """
    # The network is done training. We are ready to start on the Calibration step
    if parameterization == 'nelson_siegel':
        parameter_size = 7
        parameter_input = parameter_size + 2

    elif parameterization == 'svensson':
        parameter_size = 9
        parameter_input = parameter_size + 2

    else:
        logger.error("Unknown parameterization: %s" % parameterization)
        raise ValueError("Unknown parameterization")

    K_vector = np.array([31.6, 31.8, 32.0, 32.2, 32.4, 32.6, 32.8, 33.0, 33.2])
    tau_vector = np.array([1 / 12, 2 / 12, 3 / 12, 4 / 12, 5 / 12, 6 / 12, 1])

    # TODO: old params, new prices give with one set of old parameters to get our new parameters

    N1 = len(K_vector)
    N2 = len(tau_vector)

    calibration_grid = N1 * N2;
    grid = np.array(list(product(K_vector, tau_vector))).reshape(N1, N2, 2)
    np_input_first = np.reshape(grid, (N1 * N2, 2));  # reshape reads by rows

    parameters_to_calibrate = np.loadtxt(f'data/gridbased/gridbased_parameters_{parameterization}.dat')
    prices_calibrate = np.reshape(
            np.loadtxt(f'data/gridbased/gridbased_price_{parameterization}.dat'), newshape = (train_size + test_size,
                                                                                              N1, N2), order = 'F'
            )

    parameters_to_calibrate = parameters_to_calibrate[train_size + np.arange(test_size), :]  # size=[test_size, parameter_cal]
    prices_calibrate = prices_calibrate[train_size + np.arange(test_size), :, :]  # size=[test_size, N1, N2]

    calibration_size = parameters.reshape(-1, parameter_input).shape[0]

    # Choose optimizer and type of loss function
    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.MeanSquaredError()
    calibration_loss = tf.keras.metrics.Mean(name = 'calibration_mean')

    # This does depend on the calibration size
    calibration_dataset = tf.data.Dataset.from_tensor_slices(
            prices
            ).batch(calibration_size)

    def calibration_step(input_guess, tf_input_first, prices):

        with tf.GradientTape() as tape:
            tape.watch(input_guess)

            input_guess_rep = tf.tile(input_guess, [N1 * N2, 1])

            network_input = tf.concat([tf_input_first, input_guess_rep], axis = 1);
            prediction = model(network_input)
            c_loss = loss_object(prices, prediction)
        calibration_loss(c_loss)
        grads = tape.gradient(c_loss, [input_guess])
        optimizer.apply_gradients(zip(grads, [input_guess]))

    # We need to guess some initial model parameters. We induce errors in our old guesses here as a test
    input_guess = parameters_to_calibrate + np.random.rand(calibration_size, parameter_size) * np.array(
            [0.05]*parameter_size
            )


    # I just copy the starting parameters for convenience. This is not necessary
    old_input_guess = input_guess.copy()

    # Prepare the data to have the right shape
    tf_input_first = tf.constant(np_input_first);

    # Important: First convert to tensor, then to variable
    tf_input_guess = tf.convert_to_tensor(input_guess)

    logger.info(f"Beginning calibration for model {model_type} with {parameterization}")
    calibration_logger.info(f"Beginning calibration for model {model_type} with {parameterization}")

    # Start the actual calibration

    new_input_guess = np.zeros((calibration_size, parameter_size))
    # Start the actual calibration
    for j in tqdm(range(calibration_size)):
        np_price_local = np.reshape(prices_calibrate[j, :, :], [N1 * N2, 1])
        tf_var_input_guess_local = tf.Variable(tf.reshape(tf_input_guess[j, :], (1, parameter_size)))
        calibration_step_local = tf.function(calibration_step)
        for epoch in range(epochs):
            calibration_loss.reset_states()
            calibration_step_local(tf_var_input_guess_local, tf_input_first, np_price_local)

            template = 'Set {} Calibration Epoch {}, Loss: {}'
            message = template.format(
                    j,
                    epoch + 1,
                    calibration_loss.result(),
                    )

            calibration_logger.debug(message)

        new_input_guess[j, :] = tf_var_input_guess_local.numpy();

    change = new_input_guess - old_input_guess
    message = f"Calibration complete! change in parameters: {np.linalg.norm(change, 'fro')}"
    logger.info(message)
    calibration_logger.info(message)

    np.savetxt(f'data/pointwise/pointwise_params_calibrated_{model_type}_{parameterization}.dat', new_input_guess)
    logger.info(
            f"Saved parameters to file: "
            f"{f'data/pointwise/pointwise_params_calibrated_{model_type}_{parameterization}.dat'}"
            )

    # Errors and plots
    percentage_err = np.abs(new_input_guess - parameters_to_calibrate) / np.abs(parameters_to_calibrate)
    mean_percentage_err = np.mean(percentage_err, axis = 0) * 100
    percentage_err_copy = percentage_err.copy()
    percentage_err_copy.sort(axis = 0)
    median_percentage_err = percentage_err_copy[calibration_size // 2, :] * 100

    if plot:

        print(parameters)

        f = plt.figure(figsize = (20, 15))
        plt.subplot(3, 3, 1)
        plt.plot(parameters_to_calibrate[:, 0], percentage_err[:, 0] * 100, '*', color = 'midnightblue')
        plt.title('a')
        plt.ylabel('Percentage error');
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
        s0 = 'Average: %.2f' % mean_percentage_err[0] + r'%' + '\n' + 'Median: %.2f' % median_percentage_err[0] + r'%'
        plt.text(np.mean(parameters_to_calibrate[:, 0]), np.max(percentage_err[:, 0] * 90), s0, fontsize = 15, weight = 'bold')

        plt.subplot(3, 3, 2)
        plt.plot(parameters_to_calibrate[:, 1], percentage_err[:, 1] * 100, '*', color = 'midnightblue')
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
        plt.title('b')
        plt.ylabel('Percentage error');
        s1 = 'Average: %.2f' % mean_percentage_err[1] + r'%' + '\n' + 'Median: %.2f' % median_percentage_err[1] + r'%'
        plt.text(np.mean(parameters_to_calibrate[:, 1]), np.max(percentage_err[:, 1] * 90), s1, fontsize = 15, weight = 'bold')

        plt.subplot(3, 3, 3)
        plt.plot(parameters_to_calibrate[:, 2], percentage_err[:, 2] * 100, '*', color = 'midnightblue')
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
        plt.title('k')
        plt.ylabel('Percentage error');
        s2 = 'Average: %.2f' % mean_percentage_err[2] + r'%' + '\n' + 'Median: %.2f' % median_percentage_err[2] + r'%'
        plt.text(np.mean(parameters_to_calibrate[:, 2]), np.max(percentage_err[:, 2] * 90), s2, fontsize = 15, weight = 'bold')

        for i in range(1, parameter_size - 2):

            plt.subplot(3, 3, 3 + i)
            plt.plot(parameters_to_calibrate[:, 2 + i], percentage_err[:, 2 + i] * 100, '*', color = 'midnightblue')
            plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
            plt.title(f'$z_{i}$')
            plt.ylabel('Percentage error')
            s2 = 'Average: %.2f' % mean_percentage_err[2 + i] + r'%' + '\n' + 'Median: %.2f' % median_percentage_err[
                2 + i] + r'%'
            plt.text(np.mean(parameters_to_calibrate[:, 2 + i]), np.max(percentage_err[:, 2 + i] * 90), s2, fontsize = 15,
                     weight = 'bold')


        f.savefig(
                f'plotting/pointwise/pointwise_calibrated_{model_type}_{parameterization}.png', bbox_inches = 'tight',
                pad_inches =
                0.01
                )
