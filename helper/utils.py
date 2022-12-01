from typing import List

import numpy as np
from scipy.stats import norm

from helper.parameterizations import NelsonSiegelCurve, SvenssonCurve


class UniVolatility:
    """
    This class implements the volatility function

    Parameters
    ----------
    model_features : (array-like, 3-dimensional): [a, b, k]
        a (float, positive) : volatility coefficient
        b (float, positive) : Samuelson effect coefficient
        k (float, positive) : covariance kernel coefficient
    """

    def __init__(self, model_features: np.ndarray):
        self.model_features = model_features

    def __call__(self, tau: float, T: float, length_of_contract: float, t: float = 0) -> float:
        """This function returns the volatility parameter estimate

        Parameters
        ----------
        tau (float, positive) : time to delivery
        T (float, positive) : start of delivery
        length_of_contract (float, positive) : length of the contract
        t (float, positive) : evaluation date (default = 0)

        Returns
        -------
        volatility
        """

        a, b, k = self.model_features

        vol = (b ** 2 + 3) / 3 + ((3 * (2 - length_of_contract) * (length_of_contract ** 2) - 4) * (
                b ** 2) / 6 + 3 * length_of_contract - 2) * np.exp(-b * length_of_contract) + (
                      b ** 2 + 3) * np.exp(-2 * b * length_of_contract) / 3
        vol *= (np.exp(-2 * b * (T - tau) - np.exp(-2 * b * (T - t))))
        vol *= 2 * a ** 2 / ((length_of_contract ** 2) * (b ** 5) * k)

        return np.sqrt(vol)


class BlackScholesPrice(NelsonSiegelCurve, SvenssonCurve):

    def __init__(self, parameters: List[float], parameterization: 'nelson_siegel'):

        self.parameterization = parameterization

        if parameterization == 'nelson_siegel':
            NelsonSiegelCurve.__init__(self, parameters)

        elif parameterization == 'svensson':
            SvenssonCurve.__init__(self, parameters)

        else:
            raise ValueError(f'Parameterization {parameterization} not recognized')

    def __call__(self, option_features: np.ndarray, model_features: np.ndarray, t: float = 0) -> float:
        K, tau, T, length_of_contract = option_features

        if self.parameterization == 'nelson_siegel':
            curve = NelsonSiegelCurve(parameters = self.parameters)

        elif self.parameterization == 'svensson':
            curve = SvenssonCurve(parameters = self.parameters)

        s = UniVolatility(model_features = model_features)

        random_variable = norm()

        vol = s(tau = tau, T = T, length_of_contract = length_of_contract)
        d = (curve.calculate_average_integral_curve(lower_bound = T - t, upper_bound = T + length_of_contract - t) -
             K) \
            / vol

        result = vol * random_variable.pdf(d) + (curve.calculate_average_integral_curve(
                lower_bound = T - t, upper_bound
                = T + length_of_contract - t
                ) - K) * random_variable.cdf(d)

        return result
