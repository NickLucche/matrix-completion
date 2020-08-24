import numpy as np

def check_close_to_zero(a, dtype):
    if (dtype == np.float32):
        return np.isclose(a, [0.], atol=1e-08).item()
    elif (dtype == np.float64):
        return np.isclose(a, [0.], atol=1e-16).item()