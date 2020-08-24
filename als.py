import numpy as np
from numba import njit
from scipy import sparse
import scipy.sparse as sparse
from copy import deepcopy
from utils import check_close_to_zero
import time

class ALS:
    
    def __init__(self, u:np.ndarray, v:np.ndarray, dataset:np.ndarray ) -> None:
        # TODO: assuming u = (m, 1) v = (n, 1). (augmented)dataset: m x n, m items n users (uint8)
        self.u = u
        self.v = v
        self.X = dataset
        # compute mask from X
        self.M = (np.ones_like(dataset) * dataset).astype(np.bool)

        self.fun_eval_times = []
        self.stats = {}

    # TODO: mention all ops are vectorized for efficiency in report leveraging blas
    # TODO: specify cost of each operation
    def fit(self, eps_g:float = 1e-8, eps_params:float = 1e-10, max_iter=1000) -> np.ndarray:
        grad_u = np.ones(self.u.shape[0]) * np.inf
        grad_v = np.ones(self.v.shape[0]) * np.inf
        theta_dim = self.u.shape[0] + self.v.shape[0]
        latest_theta = np.ones(theta_dim) * np.inf
        theta = -latest_theta
        counter = 0
        start_time = time.time()

        # stopping condition: only check grad_u here since grad_v will be 0 from latest update 
        while np.linalg.norm(grad_u) > eps_g and np.linalg.norm(theta - latest_theta) > eps_params and counter < max_iter:
            latest_theta = theta
        # *** minimize wrt u ***
            # compute v_hat (masked-v) by leveraging operator broadcasting (t-th *row* will correspond to \hat{v}_t) and index-based operation
            v_hat = self.M * self.v.T
            # compute 
            self.u = _compute_minimizer(v_hat, self.X.T)
            # debug
            print(np.linalg.norm(_compute_gradient_vectorized(v_hat, self.X.T, self.u, self.v)))
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
            # debug
            print(np.linalg.norm(_compute_gradient_vectorized(np.ascontiguousarray(u_hat.T), np.asfortranarray(self.X), self.v, self.u)))

            # bookkeeping
            counter += 1
            self.log_stats(grad_u, grad_v, counter)
            print(np.linalg.norm(theta), np.linalg.norm(latest_theta))

        end_time = time.time()
        print(f"Estimated average iteration runtime: {(end_time - start_time)/counter}s")
        self.register_stats(end_time - start_time, counter, grad_u, np.linalg.norm(theta - latest_theta))
        print(f"Estimated average function evaluation times: {self.stats['avg_fun_eval_time']}")
        return self.u, self.v

    def log_stats(self, grad_u, grad_v, iteration):
        print('\nIteration\tFunction Eval (Loss)\t\tCurrent ||grad_u||\t\tPrev ||grad_v||\t')
        start = time.time()
        fun_eval = self.function_eval()
        self.fun_eval_times.append(time.time() - start)
        print(f'[{iteration}]\t\t{fun_eval}\t\t{np.linalg.norm(grad_u)}\t\t{np.linalg.norm(grad_v)}')

    def function_eval(self):
        return np.sum((self.u @ self.v.T * self.M - self.X)**2)

    def register_stats(self, runtime:float, n_iters:int, final_grad:np.ndarray, theta_diff:float):
        self.stats['avg_iter_time'] = runtime/n_iters
        self.stats['total_convergence_time'] = runtime
        self.stats['avg_fun_eval_time'] = sum(self.fun_eval_times)/len(self.fun_eval_times)
        self.stats['num_iterations'] = n_iters
        self.stats['grad_u_norm'] =  np.linalg.norm(final_grad)
        self.stats['theta_diff_norm'] = theta_diff

# TODO: explain numba, no python code here
@njit
def _compute_minimizer(hat_vect_matrix: np.ndarray, X: np.ndarray)->np.ndarray:
    # TODO: explain why it is efficient in report too (O(mn))
    # minimizer = np.sum(hat_vect_matrix*X.T , axis=1)
    # vect_of_norms = (hat_vect_matrix * hat_vect_matrix).sum(axis=1)
    # vect_of_norms[vect_of_norms<1e-8] = 1
    # return (minimizer/vect_of_norms).reshape(-1, 1)

    minimizer = np.zeros(hat_vect_matrix.shape[0]).reshape(-1, 1).astype(hat_vect_matrix.dtype)
    for i in range(hat_vect_matrix.shape[0]):
        minimizer[i] = hat_vect_matrix[i, :] @ X[:, i]#.astype(hat_vect_matrix.dtype)
        # divide by norm_2 squared
        norm = hat_vect_matrix[i] @ hat_vect_matrix[i] 
        if norm > 1e-16:
            minimizer[i] /= norm

    return minimizer

@njit
def _compute_gradient_vectorized(hat_vect_matrix: np.ndarray, X: np.ndarray, z:np.ndarray, y:np.ndarray)->np.ndarray:
    # z = z.reshape(-1, 1)
    # grad_z = (hat_vect_matrix * (z @ y.T - X.T)).sum(axis=1)
    # return grad_z.reshape(-1, 1)
    grad_z = np.zeros(z.shape[0]).reshape(-1, 1).astype(z.dtype)
    for i in range(hat_vect_matrix.shape[0]):
        # make sure X it's a col vector otherwise it will broadcast '-' operation
        grad_z[i] = hat_vect_matrix[i] @ (z[i] * y - X[:, i].reshape(-1, 1)).astype(z.dtype)

    return grad_z


class ALSSparse:
    
    def __init__(self, u:np.ndarray, v:np.ndarray, dataset: sparse.csr.csr_matrix) -> None:
        # TODO: assuming u = (m, 1) v = (n, 1). (augmented)dataset: m x n, m items n users (uint8)
        self.u = u
        self.v = v
        self.X = dataset
        # compute mask from X maintainig sparse format
        self.M = sparse.csr_matrix(dataset, dtype=np.bool)
        
        self.fun_eval_times = []
        self.stats = {}
    
    def fit(self, eps_g:float = 1e-8, eps_params:float = 1e-10, max_iter=1000) -> np.ndarray:
        grad_u = np.ones(self.u.shape[0]) * np.inf
        grad_v = np.ones(self.v.shape[0]) * np.inf
        theta_dim = self.u.shape[0] + self.v.shape[0]
        latest_theta = np.ones(theta_dim) * np.inf
        theta = -latest_theta
        counter = 0
        start_time = time.time()

        # stopping condition: only check grad_u here since grad_v will be 0 from latest update 
        while np.linalg.norm(grad_u) > eps_g and np.linalg.norm(theta - latest_theta) > eps_params and counter < max_iter:
            latest_theta = theta
        # *** minimize wrt u ***
            # compute v_hat (masked-v) by leveraging sparse representation (t-th *row* will correspond to \hat{v}_t)
            sparse_v = sparse.lil_matrix((self.v.shape[0], self.v.shape[0]), dtype=self.v.dtype)
            sparse_v.setdiag(self.v)
            v_hat = self.M * sparse_v
            # compute minimizer wrt u
            self.u = _compute_sparse_minimizer(v_hat, self.X)
            # debug
            print(np.linalg.norm(_compute_sparse_gradient_vectorized(v_hat, self.X, self.u, self.v)))
        # *** minimize wrt v ***
            # this time u_hat will have j-th *column* corresponding to \hat{u}_j since u represents users
            sparse_u = sparse.lil_matrix((self.u.shape[0], self.u.shape[0]), dtype=self.u.dtype)
            sparse_u.setdiag(self.u)
            u_hat = sparse_u * self.M     # note self.u was just minimized above
            # optional: check gradient of v
            grad_v = _compute_sparse_gradient_vectorized(u_hat.T, self.X.T, self.v, self.u)
            # compute minimizer wrt v
            self.v = _compute_sparse_minimizer(u_hat.T, self.X.T)
            
            theta = np.vstack([self.u, self.v])

            # compute gradient of u after v was updated (grad_v is zero here)
            sparse_v.setdiag(self.v)
            v_hat = self.M * sparse_v
            grad_u = _compute_sparse_gradient_vectorized(v_hat, self.X, self.u, self.v)
            
            # debug
            print(np.linalg.norm(_compute_sparse_gradient_vectorized(u_hat.T, self.X.T, self.v, self.u)))
            # bookkeeping
            counter += 1
            self.log_stats(grad_u, grad_v, counter)
            print(np.linalg.norm(theta), np.linalg.norm(latest_theta))

        end_time = time.time()
        print(f"Estimated average iteration runtime: {(end_time-start_time)/counter}s")
        self.register_stats(end_time - start_time, counter, grad_u, np.linalg.norm(theta - latest_theta))

        print(f"Estimated average function evaluation times: {self.stats['avg_fun_eval_time']}")
        
        return self.u, self.v

    def log_stats(self, grad_u, grad_v, iteration):
        print('\nIteration\tFunction Eval (Loss)\t\tCurrent ||grad_u||\t\tPrev ||grad_v||\t')
        start = time.time()
        fun_eval = self.function_eval()
        self.fun_eval_times.append(time.time() - start)
        print(f'[{iteration}]\t\t{fun_eval}\t\t{np.linalg.norm(grad_u)}\t\t{np.linalg.norm(grad_v)}')

    def function_eval(self):
        # TODO: function evaluation benefits greatly since will resolve to sparse operations-only
        return (self.M.multiply(self.u @ self.v.T) - self.X).power(2).sum()
    
    def register_stats(self, runtime:float, n_iters:int, final_grad:np.ndarray, theta_diff:float):
        self.stats['avg_iter_time'] = runtime/n_iters
        self.stats['total_convergence_time'] = runtime
        self.stats['avg_fun_eval_time'] = sum(self.fun_eval_times)/len(self.fun_eval_times)
        self.stats['num_iterations'] = n_iters
        self.stats['grad_u_norm'] =  np.linalg.norm(final_grad)
        self.stats['theta_diff_norm'] = theta_diff


def _compute_sparse_minimizer(hat_vect_matrix: sparse.csr_matrix, X: sparse.csr_matrix)->np.ndarray:
    # TODO: explain elemntwise+sum form
    # compute numerator part of minimizer, this will yield a dense vector (size of u or v)
    minimizer = (hat_vect_matrix.multiply(X)).sum(axis=1)
    # divide by norm squared of masked vector
    vector_of_norms = (hat_vect_matrix.multiply(hat_vect_matrix)).sum(axis=1)
    # if some norm is 0 numerator will be 0 too
    vector_of_norms[vector_of_norms<1e-8] = 1
    minimizer = minimizer / vector_of_norms
    # return np.ndarray representation
    return minimizer.A


def _compute_sparse_gradient_vectorized(hat_vect_matrix: sparse.csr_matrix, X: sparse.csr_matrix, z:np.ndarray, y:np.ndarray)->np.ndarray:
    # exploit product with sparse matrix, tho second term will be dense
    grad_z = (hat_vect_matrix.multiply(z @ y.T - X)).sum(axis=1)
    return grad_z.A

if __name__ == "__main__":
    u = np.random.randn(10).astype(np.float32).reshape(-1, 1)
    v = np.random.randn(3).astype(np.float32).reshape(-1, 1)
    # X = (np.random.randn(10, 3)**2).astype(np.float32)
    X = sparse.random(10, 3, density=.4, dtype=np.float32).power(2).toarray()
    hat = np.random.randn(10, 3).astype(np.float32)
    # print(_compute_minimizer(np.ascontiguousarray(hat.T), np.asfortranarray(X)))
    als = ALS(u, v, X)
    als.fit(max_iter=10)

    print('**SPARSE IMPLEMENTATION**')
    u = np.random.randn(10).astype(np.float32).reshape(-1, 1)
    v = np.random.randn(3).astype(np.float32).reshape(-1, 1)
    X_float = sparse.random(10, 3, density=.4, dtype=np.float32).power(2).tocsr() * 10
    X_int = sparse.csr_matrix(X_float, dtype=np.uint8)
    print(X_float, X_int)
    hat = np.random.randn(10, 3).astype(np.float32)
    als = ALSSparse(u, v, X_float)
    als.fit(max_iter=10)

    u = np.random.randn(10).astype(np.float32).reshape(-1, 1)
    v = np.random.randn(3).astype(np.float32).reshape(-1, 1)
    als = ALSSparse(u, v, X_int)
    als.fit(max_iter=10)


    