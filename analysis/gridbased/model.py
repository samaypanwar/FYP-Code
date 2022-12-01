"""
Choose model then check if weights present otherwise train it and save the weights

if weights are present then test / calibrate
"""
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from tqdm import tqdm

from analysis.gridbased.convolutional import CNN
from analysis.gridbased.dense import DenseModel
from analysis.gridbased import *


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

    params_range = np.loadtxt(f'data/gridbased/gridbased_parameters_{parameterization}.dat')
    price = np.reshape(
            np.loadtxt(f'data/gridbased/gridbased_price_{parameterization}.dat'),
            newshape = (train_size + test_size, N1,
                        N2),
            order = 'F'
            )

    # Train and test sets
    params_range_train = params_range[:train_size, :]  # size=[train_size, param_in]
    params_range_test = params_range[train_size: train_size + test_size, :]  # size=[test_size, param_in]

    price_train = price[:train_size, :, :]  # size=[train_size, N1, N2]
    price_test = price[train_size: train_size + test_size, :, :]  # size=[test_size, N1, N2]

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
    init_model.optimizer = tf.keras.optimizers.Adam()
    init_model.loss_object = tf.keras.losses.MeanSquaredError()

    model.compile(
            loss = init_model.loss_object, optimizer = init_model.optimizer
            )

    model.build(input_shape = (1, parameter_size))
    model.summary()

    logger.info(f"Model of type: {model_type} has been initialized")

    model(tf.ones(shape = (1, parameter_size)))

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
    path = f"weights/gridbased/gridbased_weights_{model_type}_{parameterization}.h5"

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
    # fast to the GPU. Notice that they depend on the bactch_size
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

    path_to_weights = f"weights/gridbased/gridbased_weights_{model_type}_{parameterization}.h5"
    model.save_weights(path_to_weights)
    logger.info("Saved weights to file: {}".format(path_to_weights))

    if plot:
        plot_path = 'plotting/gridbased/'
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
        plt.xlabel("Epoch", fontsize = 15, labelpad = 5)
        plt.ylabel("Loss", fontsize = 15, labelpad = 5)
        text = 'Test Loss Last Epoch = %.10f' % test_loss.result().numpy() + '\n' + 'Last Epoch = %d' % (
                epochs + 1) + '\n' + 'Batch Size = %d' % batch_size
        plt.text(epochs // 4, train_loss_vec[1] / 2, text, fontsize = 12)

        figure1.savefig(f'{plot_path}gridbased_loss_{model_type}_{parameterization}.png')

        # Heatmap train loss
        K_label = np.array([31.6, 31.8, 32.0, 32.2, 32.4, 32.6, 32.8, 33.0, 33.2])
        tau_label = ['1', '2', '3', '4', '5', '6', '12']

        figure_train = plt.figure(1, figsize = (10, 5))
        ax = plt.subplot(1, 2, 1)
        mean_err = np.mean(100 * err_training_train, axis = 0)
        plt.title("Average percentage error", fontsize = 15, y = 1.04)
        plt.imshow(np.transpose(mean_err))
        plt.colorbar(format = mtick.PercentFormatter(), pad = 0.01, fraction = 0.046)
        ax.set_xticks(np.linspace(0, N1 - 1, N1))
        ax.set_xticklabels(K_label)
        ax.set_yticks(np.linspace(0, N2 - 1, N2))
        ax.set_yticklabels(tau_label)
        plt.xlabel("Strike", fontsize = 15, labelpad = 5)
        plt.ylabel("Maturity (month)", fontsize = 15, labelpad = 5)

        ax = plt.subplot(1, 2, 2)
        max_err = np.max(100 * err_training_train, axis = 0)
        plt.title("Maximum percentage error", fontsize = 15, y = 1.04)
        plt.imshow(np.transpose(max_err))
        plt.colorbar(format = mtick.PercentFormatter(), pad = 0.01, fraction = 0.046)
        ax.set_xticks(np.linspace(0, N1 - 1, N1))
        ax.set_xticklabels(K_label)
        ax.set_yticks(np.linspace(0, N2 - 1, N2))
        ax.set_yticklabels(tau_label)
        plt.xlabel("Strike", fontsize = 15, labelpad = 5)
        plt.ylabel("Maturity (month)", fontsize = 15, labelpad = 5)

        figure_train.savefig(
                f'{plot_path}gridbased_error_train_{model_type}_{parameterization}.png', bbox_inches = 'tight',
                pad_inches = 0.01
                )

        # Heatmap test loss
        figure_test = plt.figure(1, figsize = (16, 5))
        ax = plt.subplot(1, 2, 1)
        mean_err = np.mean(100 * err_training_test, axis = 0)
        plt.title("Average percentage error", fontsize = 15, y = 1.04)
        plt.imshow(np.transpose(mean_err))
        plt.colorbar(format = mtick.PercentFormatter(), pad = 0.01, fraction = 0.046)
        ax.set_xticks(np.linspace(0, N1 - 1, N1))
        ax.set_xticklabels(K_label)
        ax.set_yticks(np.linspace(0, N2 - 1, N2))
        ax.set_yticklabels(tau_label)
        plt.xlabel("Strike", fontsize = 15, labelpad = 5)
        plt.ylabel("Maturity (month)", fontsize = 15, labelpad = 5)

        ax = plt.subplot(1, 2, 2)
        max_err = np.max(100 * err_training_test, axis = 0)
        plt.title("Maximum percentage error", fontsize = 15, y = 1.04)
        plt.imshow(np.transpose(max_err))
        plt.colorbar(format = mtick.PercentFormatter(), pad = 0.01, fraction = 0.046)
        ax.set_xticks(np.linspace(0, N1 - 1, N1))
        ax.set_xticklabels(K_label)
        ax.set_yticks(np.linspace(0, N2 - 1, N2))
        ax.set_yticklabels(tau_label)
        plt.xlabel("Strike", fontsize = 15, labelpad = 5)
        plt.ylabel("Maturity (month)", fontsize = 15, labelpad = 5)

        figure_test.savefig(
                f'{plot_path}gridbased_error_test_{model_type}_{parameterization}.png', bbox_inches =
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

    elif parameterization == 'svensson':
        parameter_size = 9

    else:
        logger.error("Unknown parameterization: %s" % parameterization)
        raise ValueError("Unknown parameterization")

    calibration_size = parameters.reshape(-1, parameter_size).shape[0]

    # Choose optimizer and type of loss function
    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.MeanSquaredError()
    calibration_loss = tf.keras.metrics.Mean(name = 'calibration_mean')

    # This does depend on the calibration size
    calibration_dataset = tf.data.Dataset.from_tensor_slices(
            prices
            ).batch(calibration_size)

    @tf.function
    def calibration_step(input_guess, prices):

        with tf.GradientTape() as tape:
            tape.watch(input_guess)
            prediction = model(input_guess)
            c_loss = loss_object(prices, prediction)
        calibration_loss(c_loss)
        grads = tape.gradient(c_loss, [input_guess])
        optimizer.apply_gradients(zip(grads, [input_guess]))

    # TODO: Instead of artificially inducing errors in our params, we can use new prices or shift prices by a bit
    # We need to guess some initial model parameters. We induce errors in our old guesses here as a test
    input_guess = parameters + np.random.rand(calibration_size, parameter_size) * np.array(
            [0.05]*parameter_size
            )

    # I just copy the starting parameters for convenience. This is not necessary
    old_input_guess = input_guess.copy()

    # Important: First convert to tensor, then to variable
    tf_input_guess = tf.convert_to_tensor(input_guess)
    tf_var_input_guess = tf.Variable(tf_input_guess)

    logger.info(f"Beginning calibration for model {model_type} with {parameterization}")
    calibration_logger.info(f"Beginning calibration for model {model_type} with {parameterization}")

    # Start the actual calibration
    for epoch in tqdm(range(epochs)):
        calibration_loss.reset_states()
        for labels in calibration_dataset:
            # For each set of labels, compute the gradient of the network, and
            # preform a gradient update on the input parameters.
            calibration_step(tf_var_input_guess, labels)

        template = 'Calibration Epoch {}, Loss: {}'
        message = template.format(
                epoch + 1,
                calibration_loss.result(),
                )

        calibration_logger.debug(message)

    new_input_guess = tf_var_input_guess.numpy()

    change = new_input_guess - old_input_guess
    message = f"Calibration complete! change in parameters: {np.linalg.norm(change, 'fro')}"
    logger.info(message)
    calibration_logger.info(message)

    np.savetxt(f'data/gridbased/gridbased_params_calibrated_{model_type}_{parameterization}.dat', new_input_guess)
    logger.info(
            f"Saved parameters to file: {f'data/gridbased/gridbased_params_calibrated_{model_type}_{parameterization}.dat'}"
            )

    # Errors and plots
    percentage_err = np.abs(new_input_guess - parameters) / np.abs(parameters)
    mean_percentage_err = np.mean(percentage_err, axis = 0) * 100
    percentage_err_copy = percentage_err.copy()
    percentage_err_copy.sort(axis = 0)
    median_percentage_err = percentage_err_copy[calibration_size // 2, :] * 100

    if plot:

        f = plt.figure(figsize = (20, 15))
        plt.subplot(3, 3, 1)
        plt.plot(parameters[:, 0], percentage_err[:, 0] * 100, '*', color = 'midnightblue')
        plt.title('a')
        plt.ylabel('Percentage error')
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
        s0 = 'Average: %.2f' % mean_percentage_err[0] + r'%' + '\n' + 'Median: %.2f' % median_percentage_err[0] + r'%'
        plt.text(np.mean(parameters[:, 0]), np.max(percentage_err[:, 0] * 90), s0, fontsize = 15, weight = 'bold')

        plt.subplot(3, 3, 2)
        plt.plot(parameters[:, 1], percentage_err[:, 1] * 100, '*', color = 'midnightblue')
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
        plt.title('b')
        plt.ylabel('Percentage error')
        s1 = 'Average: %.2f' % mean_percentage_err[1] + r'%' + '\n' + 'Median: %.2f' % median_percentage_err[1] + r'%'
        plt.text(np.mean(parameters[:, 1]), np.max(percentage_err[:, 1] * 90), s1, fontsize = 15, weight = 'bold')

        plt.subplot(3, 3, 3)
        plt.plot(parameters[:, 2], percentage_err[:, 2] * 100, '*', color = 'midnightblue')
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
        plt.title('k')
        plt.ylabel('Percentage error')
        s2 = 'Average: %.2f' % mean_percentage_err[2] + r'%' + '\n' + 'Median: %.2f' % median_percentage_err[2] + r'%'
        plt.text(np.mean(parameters[:, 2]), np.max(percentage_err[:, 2] * 90), s2, fontsize = 15, weight = 'bold')

        plt.subplot(3, 3, 4)
        plt.plot(parameters[:, 3], percentage_err[:, 3] * 100, '*', color = 'midnightblue')
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
        plt.title('$\\alpha_0$')
        plt.ylabel('Percentage error')
        s2 = 'Average: %.2f' % mean_percentage_err[3] + r'%' + '\n' + 'Median: %.2f' % median_percentage_err[3] + r'%'
        plt.text(np.mean(parameters[:, 3]), np.max(percentage_err[:, 3] * 90), s2, fontsize = 15, weight = 'bold')

        plt.subplot(3, 3, 5)
        plt.plot(parameters[:, 4], percentage_err[:, 4] * 100, '*', color = 'midnightblue')
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
        plt.title('$\\alpha_1$')
        plt.ylabel('Percentage error')
        s2 = 'Average: %.2f' % mean_percentage_err[4] + r'%' + '\n' + 'Median: %.2f' % median_percentage_err[4] + r'%'
        plt.text(np.mean(parameters[:, 4]), np.max(percentage_err[:, 4] * 90), s2, fontsize = 15, weight = 'bold')

        plt.subplot(3, 3, 6)
        plt.plot(parameters[:, 5], percentage_err[:, 5] * 100, '*', color = 'midnightblue')
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
        plt.title('$\\alpha_2$')
        plt.ylabel('Percentage error')
        s2 = 'Average: %.2f' % mean_percentage_err[5] + r'%' + '\n' + 'Median: %.2f' % median_percentage_err[5] + r'%'
        plt.text(np.mean(parameters[:, 5]), np.max(percentage_err[:, 5] * 90), s2, fontsize = 15, weight = 'bold')

        plt.subplot(3, 3, 7)
        plt.plot(parameters[:, 6], percentage_err[:, 6] * 100, '*', color = 'midnightblue')
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
        plt.title('$\\alpha_3$')
        plt.ylabel('Percentage error')
        s2 = 'Average: %.2f' % mean_percentage_err[6] + r'%' + '\n' + 'Median: %.2f' % median_percentage_err[6] + r'%'
        plt.text(np.mean(parameters[:, 6]), np.max(percentage_err[:, 6] * 90), s2, fontsize = 15, weight = 'bold')

        f.savefig(
                f'plotting/gridbased/gridbased_calibrated_{model_type}_{parameterization}.png', bbox_inches = 'tight',
                pad_inches =
                0.01
                )
