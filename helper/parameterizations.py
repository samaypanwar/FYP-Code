import numpy as np


class VasicekModel:
    """
    This class implements the analytical solution to the vasicek process PDE

    Parameters
    ----------
    parameters : vector of 4 parameters, a ,b, sigma, r
    """

    def __init__(self, parameters: np.array):
        """

        """
        self.parameters = parameters

    def __call__(self, time_to_expiry: float) -> tuple:
        """
        This function returns the value of the two parameters at the given time to expiry

        Parameters
        ----------
        time_to_expiry : time to expiry for the bond contract

        Returns
        -------
        A, C
        """

        a, b, sigma, r = self.parameters

        A = ((4 * a * b - 3 * sigma ** 2) / 4 * (b ** 3)) + time_to_expiry * (
                    (sigma ** 2 - 2 * a * b) / 2 * (b ** 2)) + np.exp(-b * time_to_expiry) * (
                        (sigma ** 2 - a * b) / b ** 3) - np.exp(-2 * b * time_to_expiry) * (sigma ** 2) / (4 * (b ** 3))

        C = (-1 / b) * (1 - np.exp(-b * time_to_expiry))

        return A, C
