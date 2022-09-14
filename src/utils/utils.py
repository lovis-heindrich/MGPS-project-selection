import numpy as np

def tau_to_sigma(tau: np.array) -> np.array:
    """ Converts the precision of a Normal distribution to its standard deviation.

    Args:
        tau (np.array): Precision values

    Returns:
        np.array: Sigma values
    """
    return np.sqrt(1/tau)