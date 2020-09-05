import numpy as np
import scipy
import os

def check_close_to_zero(a, dtype):
    if (dtype == np.float32):
        return np.isclose(a, [0.], atol=1e-08).item()
    elif (dtype == np.float64):
        return np.isclose(a, [0.], atol=1e-16).item()


def save_matrix(filename:str, M):
    if isinstance(M, np.ndarray):
        with open(f'/tmp/{filename}.npz', 'wb') as f:
            np.save(f, M)
    else:
        scipy.sparse.save_npz(f'/tmp/{filename}.npz', M.tocsr())

def load_matrix(filename:str, sparse:bool):
    if sparse:
        return scipy.sparse.load_npz(f'/tmp/{filename}.npz')
    else:
        with open(f'/tmp/{filename}.npz', 'rb') as f:
            return np.load(f)
        