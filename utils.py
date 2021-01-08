import numpy as np
import scipy
import os
import csv

def check_close_to_zero(a, dtype):
    if (dtype == np.float32):
        return np.isclose(a, [0.], atol=1e-08).item()
    elif (dtype == np.float64):
        return np.isclose(a, [0.], atol=1e-16).item()


def save_matrix(filename:str, M):
    if isinstance(M, np.ndarray):
        with open(f'/tmp/{filename}.npy', 'wb') as f:
            np.save(f, M)
    else:
        scipy.sparse.save_npz(f'/tmp/{filename}.npz', M.tocsr())

def load_matrix(filename:str, sparse:bool):
    if sparse:
        return scipy.sparse.load_npz(f'/tmp/{filename}.npz')
    else:
        if 'testX' in filename:
            return scipy.sparse.load_npz(f'/tmp/{filename}.npz')
            
        with open(f'/tmp/{filename}.npy', 'rb') as f:
            return np.load(f)

def sparse_matrix_to_csv(filename: str, X: scipy.sparse.csr_matrix):
    data, rows, cols = X.data, *X.nonzero()
    with open(filename, mode='w') as file:
        file_matrix = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for rating, user_id, movie_id in zip(data, rows, cols):
            # TODO: uint8 scaled already?
            file_matrix.writerow([user_id, movie_id, rating])



