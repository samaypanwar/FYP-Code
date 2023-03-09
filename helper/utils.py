from pathlib import Path
from typing import List

import numpy as np

from helper.parameterizations import VasicekModel


class BondPricing(VasicekModel):

    def __init__(self, parameters: List[float], parameterization: str = 'vasicek'):

        self.parameterization = parameterization

        if parameterization == 'vasicek':
            VasicekModel.__init__(self, parameters)

        else:
            raise ValueError(f'Parameterization {parameterization} not recognized')

    def __call__(self, time_to_expiry: float, coupon: float = 0, r: float = 0.02) -> float:

        if self.parameterization == 'vasicek':

            vm = VasicekModel(parameters = self.parameters)

            a, b, sigma = self.parameters

            A, C = vm(time_to_expiry = time_to_expiry, r = r)

        result = np.exp(coupon * time_to_expiry) * np.exp(A + r * C)

        return 100 * result


def assert_file_existence(path):

    filename = Path(path)
    filename.touch(exist_ok = True)

    return None


def bond_price(par_value, time_to_maturity, yield_to_maturity, coupon_rate, frequency: float = 2):

    frequency = float(frequency)
    periods = time_to_maturity * frequency
    coupon = coupon_rate * par_value / frequency
    dt = [(i+1) / frequency for i in range(int(periods))]
    price = sum([coupon / (1 + yield_to_maturity / frequency) ** (frequency * t) for t in dt]) + \
            par_value / (1 + yield_to_maturity / frequency) ** (frequency * time_to_maturity)

    return price


