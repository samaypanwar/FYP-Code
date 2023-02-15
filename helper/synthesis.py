from random import sample

import numpy as np
from scipy.stats import norm
from tqdm import tqdm

from helper.utils import BondPricing, assert_file_existence
from hyperparameters import test_size, \
    train_size


def create_features_linspace(vector_ranges: dict, num: int) -> np.array:

    a_range = np.linspace(start = vector_ranges['a'][0], stop = vector_ranges['a'][1], num = 2*num)
    b_range = np.linspace(start = vector_ranges['b'][0], stop = vector_ranges['b'][1], num = 2*num)
    sigma_range = np.linspace(start = vector_ranges['sigma'][0], stop = vector_ranges['sigma'][1], num = 2*num)
    c_range = np.linspace(start = vector_ranges['c'][0], stop = vector_ranges['c'][1], num = num)
    maturity_range = np.linspace(start = vector_ranges['maturity'][0], stop = vector_ranges['maturity'][1], num = 2*num)

    features = np.zeros((num, len(vector_ranges.keys()) + 1), dtype = np.float64)

    count = 0
    max_iter = 10 * num
    iter_num = 0
    pbar = tqdm(total = num)

    while count < num and iter_num < max_iter:

        iter_num += 1

        a = sample(a_range.tolist(), k = 1)[0]
        b = sample(b_range.tolist(), k = 1)[0]
        sigma = sample(sigma_range.tolist(), k = 1)[0]
        c = sample(c_range.tolist(), k = 1)[0]
        maturity = sample(maturity_range.tolist(), k = 1)[0]
        r = norm.rvs(loc = a / b, scale = (sigma ** 2) / (2 * b))

        # If the expected value of the interest rate is greater than 5% or the vol is greater than 80%
        if abs(a / b) < 0.05 and (sigma ** 2) / (2 * b) < 0.8 and abs(r) <= 0.1:

            features[count, :] = np.array([maturity, c, a, b, sigma, r])
            a_range = np.delete(a_range, np.where(a_range == a))
            b_range = np.delete(b_range, np.where(b_range == b))
            sigma_range = np.delete(sigma_range, np.where(sigma_range == sigma))
            c_range = np.delete(c_range, np.where(c_range == c))
            maturity_range = np.delete(maturity_range, np.where(maturity_range == maturity))

            count += 1
            pbar.update(1)

        else:

            continue
    else:
        pbar.close()

    return features, count


def generate_pointwise_data(
        parameterization: str = 'vasicek'
        ):

    # Keeps track of the range of our parameters over which we are testing
    vector_ranges = {
            'c'       : [0, 0.1],
            'maturity': [1 / 24, 20],
            # 'a' can be negative as well but we keep the parameter range as positive to keep our calculated r positive
            'a'       : [0.01, .2],
            'b'       : [1, 10],
            'sigma'   : [0.1, 1],
            }

    params_range, count = create_features_linspace(
            num = train_size + test_size, vector_ranges = vector_ranges
            )

    params_range = params_range[:count]

    price = np.empty((count, 1))

    for i in tqdm(range(count)):

        time_to_expiry, c, a, b, sigma, r = params_range[i, :]

        parameters = [a, b, sigma, r]

        bond_price = BondPricing(parameters = parameters, parameterization = parameterization)

        price[i] = bond_price(time_to_expiry = time_to_expiry, coupon = c)

    wrong_indices = np.where(price > 200)[0].tolist() or np.where(price < 70)[0].tolist()

    price = np.delete(price, wrong_indices)
    params_range = np.delete(params_range, wrong_indices, axis = 0)

    assert_file_existence(f'data/pointwise/pointwise_parameters_{parameterization}.dat')
    assert_file_existence(f'data/pointwise/pointwise_price_{parameterization}.dat')

    # The order for the columns are maturity, c, a, b, sigma, r
    np.savetxt(f'data/pointwise/pointwise_parameters_{parameterization}.dat', params_range)
    np.savetxt(f'data/pointwise/pointwise_price_{parameterization}.dat', price)

    print("Data successfully generated!")
