import os
import sys
from random import sample

import numpy as np
from scipy.stats import norm
from tqdm import tqdm

path_parent = os.path.dirname(os.getcwd())

if os.getcwd()[-8:] != 'FYP-Code':
    os.chdir(path_parent)

path = os.getcwd()
sys.path.append(path)

from helper.utils import BondPricing, assert_file_existence
from hyperparameters import test_size, \
    train_size
import argparse


def create_features_linspace(vector_ranges: dict, num: int) -> np.array:

    a_range = np.linspace(start = vector_ranges['a'][0], stop = vector_ranges['a'][1], num = 2 * num)
    b_range = np.linspace(start = vector_ranges['b'][0], stop = vector_ranges['b'][1], num = 2 * num)

    sigma_range = np.linspace(start = vector_ranges['sigma'][0], stop = vector_ranges['sigma'][1], num = 2 * num)
    eta_range = np.linspace(start = vector_ranges['eta'][0], stop = vector_ranges['eta'][1], num = 2 * num)

    rho_range = np.linspace(start = vector_ranges['rho'][0], stop = vector_ranges['rho'][1], num = num)

    c_range = np.linspace(start = vector_ranges['c'][0], stop = vector_ranges['c'][1], num = 2*num)
    maturity_range = np.linspace(
            start = vector_ranges['maturity'][0], stop = vector_ranges['maturity'][1], num = num
            )

    features = np.zeros((num, len(vector_ranges.keys()) + 2), dtype = np.float64)

    count = 0
    max_iter = 10 * num
    iter_num = 0
    pbar = tqdm(total = num, desc = 'Generating Viable Samples...')

    while count < num and iter_num < max_iter:

        iter_num += 1

        a = sample(a_range.tolist(), k = 1)[0]
        b = sample(b_range.tolist(), k = 1)[0]

        sigma = sample(sigma_range.tolist(), k = 1)[0]
        eta = sample(eta_range.tolist(), k = 1)[0]

        rho = sample(rho_range.tolist(), k = 1)[0]

        c = sample(c_range.tolist(), k = 1)[0]
        maturity = sample(maturity_range.tolist(), k = 1)[0]

        x = norm.rvs(loc = 1 / a, scale = (sigma ** 2) / (2 * a))
        y = norm.rvs(loc = 1 / b, scale = (eta ** 2) / (2 * b))

        viable_mean_rate = abs((1 / b) + (1 / a)) < 0.1
        viable_volatility = (sigma ** 2) / (2 * a) < 0.8 and (eta ** 2) / (2 * b) < 0.8
        not_large_rate = abs(x + y) <= 0.1

        # If the expected value of the interest rate is greater than 5% or the vol is greater than 80%
        if viable_volatility and viable_mean_rate and not_large_rate:

            features[count, :] = np.array([maturity, c, x, y, a, b, sigma, eta, rho])

            a_range = np.delete(a_range, np.where(a_range == a))
            b_range = np.delete(b_range, np.where(b_range == b))

            sigma_range = np.delete(sigma_range, np.where(sigma_range == sigma))
            eta_range = np.delete(eta_range, np.where(eta_range == eta))

            rho_range = np.delete(rho_range, np.where(rho_range == rho))

            c_range = np.delete(c_range, np.where(c_range == c))
            maturity_range = np.delete(maturity_range, np.where(maturity_range == maturity))

            count += 1
            pbar.update(1)

        else:
            continue
    else:
        pbar.close()

    return features, count


def generate_data(
        parameterization: str = 'two_factor'
        ):

    # Keeps track of the range of our parameters over which we are testing
    vector_ranges = {
            'c'       : [0, 0.1],
            'maturity': [1 / 24, 20],
            'a'       : [20, 5000],
            'b'       : [20, 5000],
            'sigma'   : [0.1, 1],
            'eta': [0.1, 1],
            'rho': [-1, 1]
            }

    params_range, count = create_features_linspace(
            num = train_size + test_size, vector_ranges = vector_ranges
            )

    params_range = params_range[:count]

    price = np.empty((count, 1))

    for i in tqdm(range(count), desc = 'Calculating Bond Price...'):

        time_to_expiry, c, x, y, a, b, sigma, eta, rho = params_range[i, :]

        parameters = [x, y, a, b, sigma, eta, rho]

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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
            '-p', "--parameterisation", type = str,
            help = "Parameterisation for our underlying bond pricing model", default = 'two_factor',
            choices = ['two_factor', 'vasicek']
            )

    args = parser.parse_args()

    generate_data(args.parameterisation)
