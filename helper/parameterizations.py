import numpy as np
import scipy
from numpy import exp

class TwoFactorVasicekModel:
    """
    This class implements the analytical solution to the Two factor vasicek process PDE

    Parameters
    ----------
    parameters : vector of 4 parameters, a ,b, sigma, eta, r
    """

    def __init__(self, parameters: np.array):
        """

        """
        self.parameters = parameters

    def __call__(self, T: float, t: float) -> tuple:
        """
        This function returns the value of the two parameters at the given time to expiry

        Parameters
        ----------

        Returns
        -------
        A, C
        """

        time_to_expiry = T - t

        x, y, a, b, sigma, eta, rho = self.parameters

        A = - (1 / a) * (1 - exp(-a * time_to_expiry)) * x
        B = - (1 / b) * (1 - exp(-b * time_to_expiry)) * y

        integrand_1 = scipy.integrate.quadrature(func = lambda s: (exp(-a * (T - s)) - 1) ** 2, a = t, b = T)[0]
        integrand_2 = scipy.integrate.quadrature(func = lambda s: (exp(-b * (T - s)) - 1) ** 2, a = t, b = T)[0]
        integrand_3 = scipy.integrate.quadrature(
                func = lambda s: (exp(-a * (T - s)) - 1) * (exp(-b * (T - s)) - 1), a = t, b = T
                )[0]

        C = (sigma ** 2) / (2 * np.power(a, 2)) * integrand_1
        D = (eta ** 2) / (2 * np.power(b, 2)) * integrand_2

        E = (rho * (sigma * eta) / (a * b)) * integrand_3

        return A, B, C, D, E
