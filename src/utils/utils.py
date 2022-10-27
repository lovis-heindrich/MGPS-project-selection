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

def ci(mu: float, sigma: float, z=1.959963984540054) -> tuple[float, float]:
    return (mu-sigma*z, mu+sigma*z)

def scale_normal(scale: float, dist: Normal) -> Normal:
    return Normal(scale*dist.mu, scale*dist.sigma)