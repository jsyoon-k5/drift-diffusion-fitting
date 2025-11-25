import numpy as np
from scipy import stats


def linear_normalize(w, v_min, v_max, clip=True):
    if clip:
        return np.clip((2*w - (v_min + v_max)) / (v_max - v_min), -1, 1)
    return (2*w - (v_min + v_max)) / (v_max - v_min)

def linear_denormalize(z, v_min, v_max):
    return np.clip((z * (v_max - v_min) + (v_min + v_max)) / 2, v_min, v_max)



def get_r_squared(x, y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    r_squared = r_value ** 2
    return r_squared


def np_interp_nd(x, xp, fp):
    return np.array([np.interp(x, xp, _fp) for _fp in fp.T]).T


def random_sampler(_min, _max, _min_sample=0, _max_sample=0, size=1):
    assert _min_sample + _max_sample <= 1

    den = 1 - (_min_sample + _max_sample)
    z = np.clip(np.random.uniform(-_min_sample / den, 1 + _max_sample / den, size=size), 0, 1)
    z = np.clip(z * (_max - _min) + _min, _min, _max)
    if z.size == 1:
        return z[0]
    return z


def closest_factor_pair(a):
    if a < 1:
        raise ValueError("Input must be a natural number (>= 1)")

    sqrt_a = int(np.sqrt(a))
    for i in range(sqrt_a, 0, -1):
        if a % i == 0:
            return (i, a // i)


def random_sign():
    return 1 if np.random.uniform(0, 1) > 0.5 else -1


def normalize_vector(vector, tol=1e-6):
    """Normalize the size of vector to 1."""
    norm = np.linalg.norm(vector)
    if norm <= tol: return vector
    return vector / norm


def ecdf(data):
    """
    Compute the empirical cumulative distribution function (ECDF) for given data.
    
    Args:
        data: Array-like input data
        
    Returns:
        values: Sorted unique values from the data
        cumulative_prob: Cumulative probabilities corresponding to each value
    """
    values, counts = np.unique(data, return_counts=True)
    cumulative_sum = np.cumsum(counts)
    cumulative_prob = cumulative_sum / cumulative_sum[-1]
    return values, cumulative_prob