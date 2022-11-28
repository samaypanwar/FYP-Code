import numpy as np
from tqdm import tqdm

from helper.utils import BlackScholesPrice


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


def generate_grid_data(
        train_size: int = 40_000, test_size: int = 4_000
        ):

    # Two-dimensional output sizes (the image)
    K_vector = np.linspace(start = 31.6, stop = 33.2, num = 9)
    tau_vector = np.array([1 / 12, 2 / 12, 3 / 12, 4 / 12, 5 / 12, 6 / 12, 1])

    N1 = len(K_vector)
    N2 = len(tau_vector)

    l = 1 / 12  # 1 month of delivery, fixed
    # NOTE: T_1 = tau, assumption for all the experiments

    # Keeps track of the range of our parameters over which we are testing
    vector_ranges = {
            'a' : [0.2, 0.5],
            'b' : [0.5, 0.8],
            'k' : [8, 9],
            'a0': [34.2, 34.7],
            'a1': [-1.5, -1],
            'a2': [0.2, 1.2],
            'a3': [4.5, 5]
            }

    params_range = create_features_linspace(
            num = train_size + test_size,
            vector_min = [vector_ranges[key][0] for key in vector_ranges.keys()],  # [a, b, k, a0, a1, a2, a3]
            vector_max = [vector_ranges[key][1] for key in vector_ranges.keys()]
            )

    price = np.empty((train_size + test_size, N1, N2))

    for i in tqdm(range(train_size + test_size)):

        # a, b, k
        model_features = params_range[i, :3]

        nelson_siegel_parameters = list(params_range[i, 3:])
        black_scholes_price = BlackScholesPrice(parameters = nelson_siegel_parameters)

        for j in np.arange(N1):
            K = K_vector[j]

            for k in np.arange(N2):
                price[i, j, k] = black_scholes_price(
                        option_features = np.array([K, tau_vector[k], tau_vector[k], l]),
                        model_features = model_features
                        )
    price_grid = np.reshape(price, newshape = (train_size + test_size, N1 * N2), order = 'F')

    np.savetxt('data/gridbased/gridbased_parameters_nelson_siegel.dat', params_range)
    np.savetxt('data/gridbased/gridbased_price_nelson_siegel.dat', price_grid)

    print("Data successfully generated!")


def generate_pointwise_data(
        train_size: int = 60_000, test_size: int = 6_000
        ):

    l = 1 / 12  # 1 month of delivery, fixed
    # NOTE: T_1 = tau, assumption for all the experiments

    # Keeps track of the range of our parameters over which we are testing
    vector_ranges = {
            'K'  : [31.6, 33.2],
            'tau': [1 / 12, 1],
            'a'  : [0.2, 0.5],
            'b'  : [0.5, 0.8],
            'k'  : [8, 9],
            'a0' : [34.2, 34.7],
            'a1' : [-1.5, -1],
            'a2' : [0.2, 1.2],
            'a3' : [4.5, 5]
            }

    params_range = create_features_linspace(
            num = train_size + test_size,
            vector_min = [vector_ranges[key][0] for key in vector_ranges.keys()],  # [K, tau, a, b, k, a0, a1, a2, a3]
            vector_max = [vector_ranges[key][1] for key in vector_ranges.keys()]
            )

    price = np.empty((train_size + test_size, 1))

    for i in range(train_size + test_size):

        model_features = params_range[i, 2:5]

        nelson_siegel_parameters = list(params_range[i, 5:])
        black_scholes_price = BlackScholesPrice(parameters = nelson_siegel_parameters)

        K = params_range[i, 0]
        tau = params_range[i, 1]
        price[i] = black_scholes_price(
                option_features = np.array([K, tau, tau, l]),
                model_features = model_features
                )

    np.savetxt('data/pointwise/pointwise_parameters_nelson_siegel.dat', params_range)
    np.savetxt('data/pointwise/pointwise_price_nelson_siegel.dat', price)

    print("Data successfully generated!")
