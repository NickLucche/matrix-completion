import numpy as np
from numpy.core.defchararray import count
from numpy.core.records import ndarray
from numba import njit

class ALS:
    
    def __init__(self, u:np.ndarray, v:np.ndarray, dataset:np.ndarray ) -> None:
        # TODO: assuming u = (m, 1) v = (n, 1). (augmented)dataset: m x n, m items n users (uint8)
        self.u = u
        self.v = v
        self.X = dataset
        # compute mask from X
        self.M = (np.ones_like(dataset) * dataset).astype(np.bool)

    # TODO: mention all ops are vectorized for efficiency in report leveraging blas
    # TODO: specify cost of each operation
    def fit(self, eps_g:float = 1e-8, eps_params:float = 1e-14, max_iter=1000) -> np.ndarray:
        grad_u = np.ones(self.u.shape[0]) * np.inf
        grad_v = np.ones(self.v.shape[0]) * np.inf
        theta_dim = self.u.shape[0] + self.v.shape[0]
        latest_theta = np.ones(theta_dim) * np.inf
        theta = -latest_theta
        counter = 0

        # stopping condition: only check grad_u here since grad_v will be 0 from latest update 
        while np.linalg.norm(grad_u) > eps_g and np.linalg.norm(theta - latest_theta) > eps_params and counter < max_iter:
            latest_theta = theta
        # *** minimize wrt u ***
            # compute v_hat (masked-v) by leveraging operator broadcasting (t-th *row* will correspond to \hat{v}_t) and index-based operation
            v_hat = self.M * self.v.T
            # compute 
            self.u = _compute_minimizer(v_hat, self.X.T)

        # *** minimize wrt v ***
            # this time u_hat will have j-th *column* corresponding to \hat{u}_j since u represents users
            u_hat = self.M * self.u     # note self.u was just minimized above
            # optional: check gradient of v
            grad_v = _compute_gradient_vectorized(np.ascontiguousarray(u_hat.T), np.asfortranarray(self.X), self.v, self.u)
            # compute
            # TODO: explaing contiguity and trade-off with memory
            self.v = _compute_minimizer(np.ascontiguousarray(u_hat.T), np.asfortranarray(self.X))
            
            theta = np.vstack([self.u, self.v])

            # compute gradient of u after v was updated (grad_v is zero here)
            v_hat = self.M * self.v.T
            grad_u = _compute_gradient_vectorized(v_hat, self.X.T, self.u, self.v)

            # bookkeeping
            counter += 1
            self.log_stats(grad_u, grad_v, counter)
        
        return self.u, self.v

    def log_stats(self, grad_u, grad_v, iteration):
        print('\nIteration\tCurrent grad_u\t\t\tPrev grad_v\t')
        print(f'[{iteration}]\t\t{np.linalg.norm(grad_u)}\t\t{np.linalg.norm(grad_v)}')

# TODO: explain numba, no python code here
@njit
def _compute_minimizer(hat_vect_matrix: np.ndarray, X: np.ndarray)->np.ndarray:
    # TODO: explain why it is efficient in report too (O(mn))
    minimizer = np.zeros(hat_vect_matrix.shape[0]).reshape(-1, 1).astype(hat_vect_matrix.dtype)
    for i in range(hat_vect_matrix.shape[0]):
        minimizer[i] = hat_vect_matrix[i, :] @ X[:, i] # TODO: type casting
        # divide by norm_2 squared
        minimizer[i] /= hat_vect_matrix[i] @ hat_vect_matrix[i] 

    return minimizer

@njit
def _compute_gradient_vectorized(hat_vect_matrix: np.ndarray, X: np.ndarray, z:np.ndarray, y:np.ndarray)->np.ndarray:
    grad_z = np.zeros(z.shape[0]).reshape(-1, 1).astype(z.dtype)
    for i in range(hat_vect_matrix.shape[0]):
        # make sure X it's a col vector otherwise it will broadcast '-' operation
        grad_z[i] = hat_vect_matrix[i] @ (z[i] * y - X[:, i].reshape(-1, 1)).astype(z.dtype)

    return grad_z

if __name__ == "__main__":
    u = np.arange(10).astype(np.float32).reshape(10, 1)
    v = np.arange(3).astype(np.float32).reshape(3, 1)
    X = np.random.randn(10, 3).astype(np.float32)
    hat = np.random.randn(10, 3).astype(np.float32)
    # print(_compute_minimizer(np.ascontiguousarray(hat.T), np.asfortranarray(X)))
    als = ALS(u, v, X)
    als.fit(max_iter=10)


    