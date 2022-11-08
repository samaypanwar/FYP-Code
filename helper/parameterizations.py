import numpy as np


class NelsonSiegelCurve:
    """
    This class implements the Nelson Siegel Curve algorithm

    Parameters
    ----------
    parameters : vector of 4 parameters for the Nelson-Siegel curve
    """

    def __init__(self, parameters: np.array):
        """

        """
        self.parameters = parameters

    def __call__(self, point: float) -> float:
        """
        This function returns the value of the curve at the given point

        Parameters
        ----------
        point : where the function is to be evaluated

        Returns
        -------
        function value
        """

        z1, z2, z3, z4 = self.parameters

        result = z1 + (z2 + z3 * point) * np.exp(-point * z4)

        return result

    def calculate_average_integral_curve(self, lower_bound: float, upper_bound: float) -> float:
        """
        This function calculates the averaged integral of the Nelson Siegel curve.

        Parameters
        ----------
        lower_bound : for the integral
        upper_bound : for the integral

        Returns
        -------
        Averaged value of the integral
        """

        assert lower_bound <= upper_bound

        z1, z2, z3, z4 = self.parameters
        integral_value = z1 * (upper_bound - lower_bound) +\
                         (z1 + z2 + z3 + z4 * lower_bound) *\
                         np.exp(-z4 * lower_bound) / z3 - \
                         (z1 + z2 + z3 + z4 * upper_bound) *\
                         np.exp(-z4 * upper_bound) / z3

        return integral_value / (upper_bound - lower_bound)



