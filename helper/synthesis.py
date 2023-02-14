import numpy as np
from tqdm import tqdm

from helper.utils import BondPricing, assert_file_existence
from hyperparameters import test_size, \
    train_size


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


def generate_pointwise_data(
        parameterization: str = 'two_factor'
        ):

    # Keeps track of the range of our parameters over which we are testing
    vector_ranges = {
            'c'       : [0, 0.1],
            'maturity': [1 / 24, 20],
            'x'       : [0.001, 0.1],
            'y'       : [0.001, 0.1],
            'a'       : [0.01, 0.1],
            'b'       : [0.01, 0.1],
            'sigma'   : [0.005, 0.05],
            'eta'     : [0.005, 0.05],
            'rho'     : [-1, 1],
            }

    params_range = create_features_linspace(
            num = train_size + test_size,
            vector_min = [vector_ranges[key][0] for key in sorted(vector_ranges.keys())],
            vector_max = [vector_ranges[key][1] for key in sorted(vector_ranges.keys())]
            )

    price = np.empty((train_size + test_size, 1))

    for i in tqdm(range(train_size + test_size)):

        a, b, c, eta, maturity, rho, sigma, x, y = params_range[i, :]

        parameters = [x, y, a, b, sigma, eta, rho]

        bond_price = BondPricing(parameters = parameters, parameterization = parameterization)

        price[i] = bond_price(time_to_expiry = maturity, coupon = c)

    assert_file_existence(f'data/pointwise/pointwise_parameters_{parameterization}.dat')
    assert_file_existence(f'data/pointwise/pointwise_price_{parameterization}.dat')

    # The order for the columns are maturity, c, x, y, a, b, sigma, eta, rho
    np.savetxt(
            f'data/pointwise/pointwise_parameters_{parameterization}.dat', params_range[:, [4, 2, 7, 8, 0, 1, 6,
                                                                                            3, 5]]
            )
    np.savetxt(f'data/pointwise/pointwise_price_{parameterization}.dat', price)

    print("Data successfully generated!")
