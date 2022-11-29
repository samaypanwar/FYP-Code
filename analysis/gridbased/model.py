"""
Choose model then check if weights present otherwise train it and save the weights

if weights are present then test / calibrate
"""
import os

from analysis.gridbased import *
from analysis.gridbased.convolutional import CNN
from analysis.gridbased.dense import DenseModel
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


def load_data():

    params_range = np.loadtxt('data/gridbased/gridbased_parameters_nelson_siegel.dat')
    price = np.reshape(
            np.loadtxt('data/gridbased/gridbased_price_nelson_siegel.dat'), newshape = (train_size + test_size, N1, N2),
            order = 'F'
            )

    # Train and test sets
    params_range_train = params_range[np.arange(train_size), :]  # size=[train_size, param_in]
    params_range_test = params_range[train_size + np.arange(test_size), :]  # size=[test_size, param_in]

    price_train = price[np.arange(train_size), :, :]  # size=[train_size, N1, N2]
    price_test = price[train_size + np.arange(test_size), :, :]  # size=[test_size, N1, N2]

    return params_range_train, params_range_test, price_train, price_test


def init_model(model_type: str = 'dense'):

    # Create an instance of the model
    if model_type.lower() == 'dense':
        model = DenseModel()

    elif model_type.lower() == 'cnn':
        model = CNN()

    else:
        raise ValueError("Unknown model type")

    # Choose optimizer and type of loss function
    init_model.optimizer = tf.keras.optimizers.Adam()
    init_model.loss_object = tf.keras.losses.MeanSquaredError()

    model.compile(
            loss = init_model.loss_object, optimizer = init_model.optimizer
            )
    model.build(input_shape = (1, param_in))
    model.summary()

    model(tf.ones(shape = (1, param_in)))

    return model


def load_weights(model, model_type: str = 'dense', parameterization: str = 'nelson_siegel'):

    path = f"weights/gridbased/gridbased_weights_{model_type}_{parameterization}.h5"

    if os.path.isfile(path):
        model.load_weights(path)

    else:
        # TODO: logging here
        raise ValueError("Please train the weights. No such file or directory {0}".format(path))


def train_model(
        model, epochs: int = 200, batch_size: int = 30, patience: int = 20, delta: int = 0.0002,
        model_type: str = 'dense', parameterization: str = 'nelson_siegel', plot: bool = True
        ):

    params_range_train, params_range_test, price_train, price_test = load_data()

    # Choose what type of information you want to print
    train_loss = tf.keras.metrics.Mean(name = 'train_mean')
    test_loss = tf.keras.metrics.Mean(name = 'test_mean')

    # To speed up training we need to create a some object which can send the data
    # fast to the GPU. Notice that they depend on the bactch_size
    train_ds = tf.data.Dataset.from_tensor_slices(
            (params_range_train, price_train)
            ).shuffle(10000).batch(batch_size)

    test_ds = tf.data.Dataset.from_tensor_slices((params_range_test, price_test)).batch(batch_size)

    # Define the early stop function
    def early_stop(loss_vec):
        delta_loss = np.abs(np.diff(loss_vec))
        delta_loss = delta_loss[-patience:]
        return np.prod(delta_loss < delta)

    # Next we compile a few low level functions which will compute the actual gradient.

    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.MeanSquaredError()

    @tf.function
    def train_step(input_param, prices):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = model(input_param, training = True)
            loss = loss_object(prices, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)

    @tf.function
    def test_step(input_param, prices):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(input_param, training = False)
        t_loss = model.loss(prices, predictions)

        test_loss(t_loss)

    # Vectors of loss
    test_loss_vec = np.array([0.0])
    train_loss_vec = np.array([0.0])

    # We start to train the network.
    # TODO: add logging for this
    print('\nStarting to train')

    for epoch in range(epochs):

        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        test_loss.reset_states()

        for input_param, prices in train_ds:
            # For each batch of images, and prices, compute the gradient and update
            # the gradient of the network
            train_step(input_param, prices)

        for test_images, test_prices in test_ds:
            # For each of the test data, compute how the network performs on the data
            # We do not compute any gradients here
            test_step(test_images, test_prices)

        # Print some usfull information
        template = 'Epoch {0}/{1}, Loss: {2:.10f},  Test Loss: {3:.10f}, Delta Test Loss: {4:.10f}'
        print(
                template.format(
                        epoch + 1,
                        epochs,
                        train_loss.result().numpy(),
                        test_loss.result().numpy(),
                        np.abs(test_loss.result() - test_loss_vec[-1])
                        )
                )

        train_loss_vec = np.append(train_loss_vec, train_loss.result())
        test_loss_vec = np.append(test_loss_vec, test_loss.result())

        if epoch > patience and early_stop(test_loss_vec):
            print('Early stopping at epoch = ', epoch + 1)
            break

    path_to_weights = f"weights/gridbased/gridbased_weights_{model_type}_{parameterization}.h5"
    # TODO: add logging for this
    model.save_weights(path_to_weights)

    if plot:
        plot_path = 'plotting/gridbased/'
        price_predicted_train = model(params_range_train).numpy()
        price_predicted_test = model(params_range_test).numpy()

        err_training_train = abs(price_predicted_train - price_train) / price_train
        err_training_test = abs(price_predicted_test - price_test) / price_test

        # Loss plots
        l2 = plt.figure(figsize = (14, 5))
        plt.subplot(1, 2, 1)
        plt.plot(np.arange(epochs + 1), train_loss_vec[1:], '-g')
        plt.plot(np.arange(epochs + 1), test_loss_vec[1:], '-m')
        plt.legend(['Training Loss', 'Test Loss'])
        plt.xlabel("Epoch", fontsize = 15, labelpad = 5);
        plt.ylabel("Loss", fontsize = 15, labelpad = 5);
        text = 'Test Loss Last Epoch = %.10f' % test_loss.result().numpy() + '\n' + 'Last Epoch = %d' % (
                    epochs + 1) + '\n' + 'Batch Size = %d' % batch_size
        plt.text(epochs // 4, train_loss_vec[1] / 2, text, fontsize = 12);

        plt.subplot(1, 2, 2)
        plt.plot(np.arange(epochs - 50, epochs + 1), train_loss_vec[(epochs - 50 + 1):], '-g')
        plt.plot(np.arange(epochs - 50, epochs + 1), test_loss_vec[(epochs - 50 + 1):], '-m')
        plt.legend(['Training Loss', 'Test Loss'])
        plt.xlabel("Epoch", fontsize = 15, labelpad = 5);
        plt.ylabel("Loss", fontsize = 15, labelpad = 5);

        l2.savefig(f'{plot_path}gridbased_loss_step1_{model_type}_{parameterization}.pdf')

        # Heatmap train loss
        K_label = np.array([31.6, 31.8, 32.0, 32.2, 32.4, 32.6, 32.8, 33.0, 33.2])
        tau_label = ['1', '2', '3', '4', '5', '6', '12']

        f_train = plt.figure(1, figsize = (16, 5))
        ax = plt.subplot(1, 2, 1)
        mean_err = np.mean(100 * err_training_train, axis = 0)
        plt.title("Average percentage error", fontsize = 15, y = 1.04)
        plt.imshow(np.transpose(mean_err))
        plt.colorbar(format = mtick.PercentFormatter(), pad = 0.01, fraction = 0.046)
        ax.set_xticks(np.linspace(0, N1 - 1, N1))
        ax.set_xticklabels(K_label)
        ax.set_yticks(np.linspace(0, N2 - 1, N2))
        ax.set_yticklabels(tau_label)
        plt.xlabel("Strike", fontsize = 15, labelpad = 5);
        plt.ylabel("Maturity (month)", fontsize = 15, labelpad = 5);

        ax = plt.subplot(1, 2, 2)
        max_err = np.max(100 * err_training_train, axis = 0)
        plt.title("Maximum percentage error", fontsize = 15, y = 1.04)
        plt.imshow(np.transpose(max_err))
        plt.colorbar(format = mtick.PercentFormatter(), pad = 0.01, fraction = 0.046)
        ax.set_xticks(np.linspace(0, N1 - 1, N1))
        ax.set_xticklabels(K_label)
        ax.set_yticks(np.linspace(0, N2 - 1, N2))
        ax.set_yticklabels(tau_label)
        plt.xlabel("Strike", fontsize = 15, labelpad = 5);
        plt.ylabel("Maturity (month)", fontsize = 15, labelpad = 5);

        f_train.savefig(f'{plot_path}gridbased_error_step1_train_{model_type}_{parameterization}.pdf', bbox_inches = 'tight', pad_inches = 0.01)

        # Heatmap test loss
        f_test = plt.figure(1, figsize = (16, 5))
        ax = plt.subplot(1, 2, 1)
        mean_err = np.mean(100 * err_training_test, axis = 0)
        plt.title("Average percentage error", fontsize = 15, y = 1.04)
        plt.imshow(np.transpose(mean_err))
        plt.colorbar(format = mtick.PercentFormatter(), pad = 0.01, fraction = 0.046)
        ax.set_xticks(np.linspace(0, N1 - 1, N1))
        ax.set_xticklabels(K_label)
        ax.set_yticks(np.linspace(0, N2 - 1, N2))
        ax.set_yticklabels(tau_label)
        plt.xlabel("Strike", fontsize = 15, labelpad = 5);
        plt.ylabel("Maturity (month)", fontsize = 15, labelpad = 5);

        ax = plt.subplot(1, 2, 2)
        max_err = np.max(100 * err_training_test, axis = 0)
        plt.title("Maximum percentage error", fontsize = 15, y = 1.04)
        plt.imshow(np.transpose(max_err))
        plt.colorbar(format = mtick.PercentFormatter(), pad = 0.01, fraction = 0.046)
        ax.set_xticks(np.linspace(0, N1 - 1, N1))
        ax.set_xticklabels(K_label)
        ax.set_yticks(np.linspace(0, N2 - 1, N2))
        ax.set_yticklabels(tau_label)
        plt.xlabel("Strike", fontsize = 15, labelpad = 5);
        plt.ylabel("Maturity (month)", fontsize = 15, labelpad = 5);

        f_test.savefig(f'{plot_path}gridbased_error_step1_test_{model_type}_{parameterization}.pdf', bbox_inches =
        'tight', pad_inches = 0.01)

