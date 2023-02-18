"""
This file contains the training function implementation for our model
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tqdm import tqdm

from analysis.pointwise import logger, training_logger
from analysis.pointwise.utils import load_data
from helper.utils import assert_file_existence
from hyperparameters import coupon_range, maturities, maturities_label, number_of_coupon_rates, number_of_maturities


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
    figure1, ax = plt.subplots(figsize = (8, 7))
    ax.plot(np.arange(min(epochs, epoch + 1)), train_loss_vec[1:], '-g')
    ax.plot(np.arange(min(epochs, epoch + 1)), test_loss_vec[1:], '-m')
    ax.legend(['Training Loss', 'Test Loss'])

    ax.set_xlabel("Epoch", fontsize = 15, labelpad = 5);
    ax.set_ylabel("Loss", fontsize = 15, labelpad = 5);
    text = 'Test Loss Last Epoch = %.10f' % test_loss.result().numpy() + '\n' + 'Last Epoch = %d' % (
            epochs + 1) + '\n' + 'Batch Size = %d' % batch_size
    ax.text(min(epochs, epoch + 1) // 4, train_loss_vec[1] / 2, text, fontsize = 12);

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
            pad_inches = 0.3
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
            'tight', pad_inches = 0.3
            )
