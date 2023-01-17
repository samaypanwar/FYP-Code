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

    def __call__(self, time_to_expiry: float) -> float:

        if self.parameterization == 'vasicek':

            vm = VasicekModel(parameters = self.parameters)

            a, b, sigma, r = self.parameters

            A, C = vm(time_to_expiry = time_to_expiry)

        result = np.exp(A + r * C)

        return result


def assert_file_existence(path):

    filename = Path(path)
    filename.touch(exist_ok = True)

    return None
