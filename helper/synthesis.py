import numpy as np
from tqdm import tqdm

from helper.utils import BondPricing, assert_file_existence
from hyperparameters import coupon_range, maturities, number_of_coupon_rates, number_of_maturities, test_size, \
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


def generate_grid_data(
        parameterization: str = 'vasicek'
        ):

    vector_ranges = {
            'a'    : [0.01, 0.1],
            'b'    : [0.25, 0.8],
            'sigma': [0.005, 0.05],
            'r'    : [0.005, 0.06]
            }

    params_range = create_features_linspace(
            num = train_size + test_size,
            vector_min = [vector_ranges[key][0] for key in sorted(vector_ranges.keys())],
            vector_max = [vector_ranges[key][1] for key in sorted(vector_ranges.keys())]
            )

    price = np.empty((train_size + test_size, number_of_maturities, number_of_coupon_rates))

    for i in tqdm(range(train_size + test_size)):

        parameters = list(params_range[i, :])

        bond_price = BondPricing(parameters = parameters, parameterization = parameterization)

        for maturity_idx, maturity in enumerate(maturities):
            for coupon_idx, coupon in enumerate(coupon_range):

                price[i, maturity_idx, coupon_idx] = bond_price(
                        time_to_expiry = maturity,
                        coupon = coupon
                        )

    price_grid = np.reshape(
            price, newshape = (train_size + test_size, number_of_maturities * number_of_coupon_rates),
            order = 'F'
            )

    assert_file_existence(f'data/gridbased/gridbased_parameters_{parameterization}.dat')
    assert_file_existence(f'data/gridbased/gridbased_price_{parameterization}.dat')

    np.savetxt(f'data/gridbased/gridbased_parameters_{parameterization}.dat', params_range)
    np.savetxt(f'data/gridbased/gridbased_price_{parameterization}.dat', price_grid)

    print("Data successfully generated!")


def generate_pointwise_data(
        parameterization: str = 'vasicek'
        ):

    # Keeps track of the range of our parameters over which we are testing
    vector_ranges = {
            'c'       : [0, 0.1],
            'maturity': [1 / 24, 20],
            'a'       : [0.01, 0.1],
            'b'       : [0.25, 0.8],
            'sigma'   : [0.005, 0.05],
            'r'       : [0.005, 0.06]
            }

    params_range = create_features_linspace(
            num = train_size + test_size,
            vector_min = [vector_ranges[key][0] for key in sorted(vector_ranges.keys())],
            vector_max = [vector_ranges[key][1] for key in sorted(vector_ranges.keys())]
            )

    price = np.empty((train_size + test_size, 1))

    for i in tqdm(range(train_size + test_size)):

        a, b, c, time_to_expiry, r, sigma = params_range[i, :]

        parameters = [a, b, sigma, r]

        bond_price = BondPricing(parameters = parameters, parameterization = parameterization)

        price[i] = bond_price(time_to_expiry = time_to_expiry, coupon = c)

    assert_file_existence(f'data/pointwise/pointwise_parameters_{parameterization}.dat')
    assert_file_existence(f'data/pointwise/pointwise_price_{parameterization}.dat')

    # The order for the columns are maturity, c, a, b, sigma, r
    np.savetxt(f'data/pointwise/pointwise_parameters_{parameterization}.dat', params_range[:, [3, 2, 0, 1, 5, 4]])
    np.savetxt(f'data/pointwise/pointwise_price_{parameterization}.dat', price)

    print("Data successfully generated!")
