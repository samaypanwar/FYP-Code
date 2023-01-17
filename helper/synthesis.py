import numpy as np
from tqdm import tqdm

from helper.utils import BondPricing, assert_file_existence


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
        train_size: int = 40_000, test_size: int = 4_000, parameterization: str = 'vasicek'
        ):


    # TODO: GET FROM MARKET DATA MATURITIES
    maturities = np.array([1 / 12, 2 / 12, 3 / 12, 4 / 12, 5 / 12, 6 / 12, 1, 2, 3, 5, 10, 20, 30])

    N1 = len(maturities)

    vector_ranges = {
            'a'    : [0.01, 0.1],
            'b'    : [0.25, 0.8],
            'sigma': [0.1, 0.2],
            'r'    : [0.001, 0.06]
            }

    params_range = create_features_linspace(
            num = train_size + test_size,
            vector_min = [vector_ranges[key][0] for key in vector_ranges.keys()],  # [a, b, k, a0, a1, a2, a3]
            vector_max = [vector_ranges[key][1] for key in vector_ranges.keys()]
            )

    price = np.empty((train_size + test_size, N1))

    for i in tqdm(range(train_size + test_size)):

        parameters = list(params_range[i, :])
        bond_price = BondPricing(parameters = parameters, parameterization = parameterization)

        for maturity in np.arange(N1):

            price[i, maturity] = bond_price(
                    time_to_expiry = maturity
                    )

    price_grid = np.reshape(price, newshape = (train_size + test_size, N1), order = 'F')

    assert_file_existence(f'data/gridbased/gridbased_parameters_{parameterization}.dat')
    assert_file_existence(f'data/gridbased/gridbased_price_{parameterization}.dat')

    np.savetxt(f'data/gridbased/gridbased_parameters_{parameterization}.dat', params_range)
    np.savetxt(f'data/gridbased/gridbased_price_{parameterization}.dat', price_grid)

    print("Data successfully generated!")


def generate_pointwise_data(
        train_size: int = 40_000, test_size: int = 4_000, parameterization: str = 'vasicek'
        ):

    # Keeps track of the range of our parameters over which we are testing
    vector_ranges = {
            'maturity': [1 / 12, 30],
            'a'    : [0.01, 0.1],
            'b'    : [0.25, 0.8],
            'sigma': [0.1, 0.2],
            'r'    : [0.001, 0.06]
            }

    params_range = create_features_linspace(
            num = train_size + test_size,
            vector_min = [vector_ranges[key][0] for key in vector_ranges.keys()],
            vector_max = [vector_ranges[key][1] for key in vector_ranges.keys()]
            )

    price = np.empty((train_size + test_size, 1))

    for i in tqdm(range(train_size + test_size)):

        parameters = list(params_range[i, 1:])
        bond_price = BondPricing(parameters = parameters, parameterization = parameterization)

        time_to_expiry = params_range[i, 0]

        price[i] = bond_price(time_to_expiry = time_to_expiry)

    np.savetxt(f'data/pointwise/pointwise_parameters_{parameterization}.dat', params_range)
    np.savetxt(f'data/pointwise/pointwise_price_{parameterization}.dat', price)

    print("Data successfully generated!")
