""" 
Utility functions for working with probabilities.
"""

import numpy as np
from src.utils.distributions import Normal

def tau_to_sigma(tau: np.ndarray) -> np.ndarray:
    """ Converts the precision of a Normal distribution to its standard deviation.

    Args:
        tau (np.array): Precision values

    Returns:
        np.array: Sigma values
    """
    return np.sqrt(1/tau)

def sigma_to_tau(sigma: np.ndarray) -> np.ndarray:
    """ Converts the precision of a Normal distribution to its standard deviation.

    Args:
        tau (np.array): Precision values

    Returns:
        np.array: Sigma values
    """
    return 1/np.square(sigma)

def ci(mu: float, sigma: float, z=1.959963984540054) -> tuple[float, float]:
    """ Computes the confidence interval of a Normal distribution.

    Args:
        mu (float): Mean
        sigma (float): Standard deviation
        z (float, optional): Standard deviations from the mean. Defaults to 1.959963984540054 (95% confidence interval).

    Returns:
        tuple[float, float]: Bounds of the confidence interval
    """
    return (mu-sigma*z, mu+sigma*z)

def scale_normal(scale: float, dist: Normal) -> Normal:
    """ Scales a Normal distribution by a constant factor

    Args:
        scale (float): Scaling factor
        dist (Normal): Normal distribution

    Returns:
        Normal: Normal distribution N(scale*mu, scale*sigma)
    """
    return Normal(scale*dist.mu, scale*dist.sigma)