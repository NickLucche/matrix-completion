import numpy as np
from numba import njit, prange
from scipy import sparse
import scipy.sparse as sparse
from utils import check_close_to_zero
import time


class ALS:

    def __init__(self, u: np.ndarray, v: np.ndarray, dataset: np.ndarray) -> None:
        """Alternating Least Squares using standard numpy vectors. 
        Args:
            u (np.ndarray): (mx1) users (column) vector
            v (np.ndarray): (vx1) items (column) vector
            dataset (np.ndarray): (mxn) augmented dataset, full matrix with zeros representing missing values.
        """

        self.u = u
        self.v = v
        self.X = dataset
        # compute mask from X
        self.M = dataset.astype(np.bool_)

        self.fun_eval_times = []
        self.fun_evaluations = []
        self.grad_theta = []
        self.stats = {}

    def fit(self, eps_g: float = 1e-8, eps_params: float = 1e-10, max_iter=1000) -> np.ndarray:
        grad_u = np.ones(self.u.shape[0]) * np.inf
        grad_v = np.ones(self.v.shape[0]) * np.inf
        theta_dim = self.u.shape[0] + self.v.shape[0]
        latest_theta = np.ones(theta_dim) * np.inf
        theta = -latest_theta
        counter = 0
        norm_grad_u = np.linalg.norm(grad_u)
        start_time = time.time()

        # stopping condition: only check grad_u here since grad_v will be 0 from latest update 
        # FIXME: norm here uses too much memory (weird)
        while norm_grad_u > eps_g and np.linalg.norm(
                theta - latest_theta) > eps_params and counter < max_iter:
            latest_theta = theta
        # *** minimize wrt u ***
            # V_hat (t-th *row* will correspond to \hat{v}_t=M[t]*v), computed inside to save memory
            # compute minimizer for u
            self.u = _compute_minimizer(self.v.T, self.M, self.X.T)

            # debug, grad should be ~0 here
            # print(np.linalg.norm(_compute_gradient_vectorized(v_hat, self.X.T, self.u, self.v)))

        # *** minimize wrt v ***
            # this time u_hat will have j-th *column* corresponding to \hat{u}_j since u represents users
            # optional: check gradient of v
            grad_v = _compute_gradient_vectorized(self.M.T, np.asfortranarray(self.X), self.v, self.u)
            

            # TODO: explaing contiguity and trade-off with memory
            self.v = _compute_minimizer(self.u.T, self.M.T, np.asfortranarray(self.X))

            theta = np.vstack([self.u, self.v])

            # compute gradient of u after v was updated (grad_v is zero here)
            # v_hat = self.M * self.v.T computed inside
            grad_u = _compute_gradient_vectorized(self.M, self.X.T, self.u, self.v)
            
            norm_grad_u = np.linalg.norm(grad_u)
            self.grad_theta.append(norm_grad_u)
            
            # debug, grad should be ~0 here
            # print(np.linalg.norm(_compute_gradient_vectorized(np.ascontiguousarray(u_hat.T), np.asfortranarray(self.X), self.v, self.u)))

            # bookkeeping
            counter += 1
            self.log_stats(norm_grad_u, grad_v, counter)

            # debug: checking difference in thetas
            # print(np.linalg.norm(theta), np.linalg.norm(latest_theta))

        end_time = time.time()
        print('-'*80)
        print(f"Estimated average iteration runtime: {(end_time - start_time) / counter}s")
        self.register_stats(end_time - start_time, counter, grad_u, np.linalg.norm(theta - latest_theta))
        print(f"Estimated average function evaluation times: {self.stats['avg_fun_eval_time']}")
        return self.u, self.v

    def log_stats(self, norm_grad_u, grad_v, iteration):
        print('\nIteration\tFunction Eval (Loss)\t\tCurrent ||grad_u||\t\tPrev ||grad_v||\t')
        start = time.time()
        fun_eval = self.function_eval()
        self.fun_eval_times.append(time.time() - start)
        self.fun_evaluations.append(fun_eval)
        print(f'[{iteration}]\t\t{fun_eval}\t\t{norm_grad_u}\t\t{np.linalg.norm(grad_v)}')

    def function_eval(self):
        # avoid computing uv^T directly for saving memory
        # return np.sum((self.u @ self.v.T * self.M - self.X)**2)
        return _function_eval_numba(self.X, self.u, self.v, self.M)

    def register_stats(self, runtime: float, n_iters: int, final_grad: np.ndarray, theta_diff: float):
        self.stats['avg_iter_time'] = runtime / n_iters
        self.stats['total_convergence_time'] = runtime
        self.stats['avg_fun_eval_time'] = sum(self.fun_eval_times) / len(self.fun_eval_times)
        self.stats['num_iterations'] = n_iters
        self.stats['grad_u_norm'] = np.linalg.norm(final_grad)
        self.stats['theta_diff_norm'] = theta_diff
        self.stats['mse'] = self.function_eval() / np.count_nonzero(self.X)
        self.stats['fun_evals'] = self.fun_evaluations
        self.stats['grad_theta'] = self.grad_theta


@njit(parallel=False)
def _function_eval_numba(X: np.ndarray, u: np.ndarray, v: np.ndarray, M: np.ndarray):
    rows_sum = np.zeros(X.shape[1], dtype=u.dtype).reshape(-1, 1)
    for i in range(X.shape[0]):
        rows_sum[i] = (((u[i].item() * v) * M[i].reshape(-1, 1) - X[i].reshape(-1, 1)) ** 2).sum()
    return rows_sum.sum()


@njit(parallel=False)
def _compute_minimizer(z: np.ndarray, M:np.ndarray, X: np.ndarray) -> np.ndarray:
    # minimizer = np.sum(hat_vect_matrix*X.T , axis=1)   <--- explicit but memory inefficient form
    # vect_of_norms = (hat_vect_matrix * hat_vect_matrix).sum(axis=1)
    # vect_of_norms[vect_of_norms<1e-8] = 1
    # return (minimizer/vect_of_norms).reshape(-1, 1)

    minimizer = np.zeros(X.shape[1]).reshape(-1, 1).astype(z.dtype)
    for i in prange(X.shape[1]):
        # row vector (i-th row of V_hat/U_hat)
        masked_vector = M[i] * z
        minimizer[i] = masked_vector @ X[:, i].astype(z.dtype)
        # divide by norm_2 squared of hat vector
        norm = (masked_vector @ masked_vector.T).item()
        if norm > 1e-16:
            minimizer[i] /= norm

    return minimizer


@njit(parallel=False)
def _compute_gradient_vectorized(M: np.ndarray, X: np.ndarray, z: np.ndarray,
                                 y: np.ndarray) -> np.ndarray:
    # z = z.reshape(-1, 1)
    # grad_z = (hat_vect_matrix * (z @ y.T - X.T)).sum(axis=1)  <--- explicit but memory inefficient form
    # return grad_z.reshape(-1, 1)
    grad_z = np.zeros(z.shape[0]).reshape(-1, 1).astype(z.dtype)
    for i in prange(z.shape[0]):
        hat_vector = M[i] * y.T
        # make sure X it's a col vector otherwise it will broadcast '-' operation
        grad_z[i] = hat_vector @ (z[i] * y - X[:, i].reshape(-1, 1).astype(z.dtype)).astype(z.dtype)

    return grad_z


class ALSSparse:

    def __init__(self, u: np.ndarray, v: np.ndarray, dataset: sparse.csr.csr_matrix) -> None:
        """Alternating Least Squares using scipy-based sparse vectors implementation. 
        Args:
            u (np.ndarray): (mx1) users (column) vector
            v (np.ndarray): (nx1) items (column) vector
            dataset (np.ndarray): (mxn) sparse dataset of ratings.
        """
        self.u = u
        self.v = v
        self.X = dataset
        # need for computing sparse gradient efficiently (just a reference)
        self.X_T = self.X.tocsc(copy=False).T
        # compute mask from X maintainig sparse format
        self.M = sparse.csr_matrix(dataset, dtype=np.bool)

        self.fun_eval_times = []
        self.fun_evaluations = []
        self.grad_theta = []
        self.stats = {}

    def fit(self, eps_g: float = 1e-8, eps_params: float = 1e-10, max_iter=1000) -> np.ndarray:
        grad_u = np.ones(self.u.shape[0]) * np.inf
        grad_v = np.ones(self.v.shape[0]) * np.inf
        theta_dim = self.u.shape[0] + self.v.shape[0]
        latest_theta = np.ones(theta_dim) * np.inf
        theta = -latest_theta
        counter = 0
        norm_grad_u = np.linalg.norm(grad_u)
        start_time = time.time()

        # stopping condition: only check grad_u here since grad_v will be 0 from latest update 
        while norm_grad_u > eps_g and np.linalg.norm(
                theta - latest_theta) > eps_params and counter < max_iter:
            latest_theta = theta
        # *** minimize wrt u ***
            # compute v_hat (masked-v) by leveraging sparse representation (t-th *row* will correspond to \hat{v}_t)
            sparse_v = sparse.lil_matrix((self.v.shape[0], self.v.shape[0]), dtype=self.v.dtype)
            # build a sparse matrix from v then use matrix mul (*) to compute M \odot V (elemtnwise product, broadcasted) efficiently
            sparse_v.setdiag(self.v)
            v_hat = self.M * sparse_v
            # compute minimizer wrt u
            self.u = _compute_sparse_minimizer(v_hat, self.X)

            # debug, grad should be ~0 here
            # print(np.linalg.norm(_compute_sparse_gradient(v_hat, self.X, self.u, self.v)))

        # *** minimize wrt v ***
            # this time u_hat will have j-th *column* corresponding to \hat{u}_j since u represents users
            sparse_u = sparse.lil_matrix((self.u.shape[0], self.u.shape[0]), dtype=self.u.dtype)
            sparse_u.setdiag(self.u)
            u_hat = sparse_u * self.M  # note self.u was just minimized above
            # optional: check gradient of v (X to X.T is cheap in sparse matrices, csr->csc)
            grad_v = _compute_sparse_gradient(u_hat.T, self.X_T, self.v, self.u)

            # compute minimizer wrt v
            self.v = _compute_sparse_minimizer(u_hat.T, self.X.T)

            theta = np.vstack([self.u, self.v])

            # compute gradient of u after v was updated (grad_v is zero here)
            sparse_v.setdiag(self.v)
            v_hat = self.M * sparse_v
            grad_u = _compute_sparse_gradient(v_hat, self.X, self.u, self.v)

            # keep track of grad
            norm_grad_u = np.linalg.norm(grad_u)
            self.grad_theta.append(norm_grad_u) 

            # debug, grad should be ~0 here
            # print(np.linalg.norm(_compute_sparse_gradient_vectorized(u_hat.T, self.X.T, self.v, self.u)))

            # bookkeeping
            counter += 1
            self.log_stats(norm_grad_u, grad_v, counter)

            # debug: checking difference in thetas
            # print(np.linalg.norm(theta), np.linalg.norm(latest_theta))

        end_time = time.time()
        print('-'*80)
        print(f"Estimated average iteration runtime: {(end_time - start_time) / counter}s")
        self.register_stats(end_time - start_time, counter, grad_u, np.linalg.norm(theta - latest_theta))

        print(f"Estimated average function evaluation times: {self.stats['avg_fun_eval_time']}")

        return self.u, self.v

    def log_stats(self, norm_grad_u, grad_v, iteration):
        print('\nIteration\tFunction Eval (Loss)\t\tCurrent ||grad_u||\t\tPrev ||grad_v||\t')
        start = time.time()
        fun_eval = self.function_eval()
        self.fun_eval_times.append(time.time() - start)
        self.fun_evaluations.append(fun_eval)
        print(f'[{iteration}]\t\t{fun_eval}\t\t{norm_grad_u}\t\t{np.linalg.norm(grad_v)}')

    def function_eval(self):
        # vectorized form requires more memory (compute `A=uv^T`)
        # return (self.M.multiply(self.u @ self.v.T) - self.X).power(2).sum()
        # efficient sparse computation since there's no need to multiply by M (uv^T-X avoids nonzero elems by construction) 
        sparse_X_tuple = (self.X.data, *self.X.nonzero())
        differences, _ = _compute_sparse_difference_matrix(sparse_X_tuple, self.u, self.v)
        return (np.array(differences, dtype=self.u.dtype) ** 2).sum()

    def register_stats(self, runtime: float, n_iters: int, final_grad: np.ndarray, theta_diff: float):
        self.stats['avg_iter_time'] = runtime / n_iters
        self.stats['total_convergence_time'] = runtime
        self.stats['avg_fun_eval_time'] = sum(self.fun_eval_times) / len(self.fun_eval_times)
        self.stats['num_iterations'] = n_iters
        self.stats['grad_u_norm'] = np.linalg.norm(final_grad)
        self.stats['theta_diff_norm'] = theta_diff
        self.stats['mse'] = self.function_eval() / self.X.getnnz()
        self.stats['fun_evals'] = self.fun_evaluations
        self.stats['grad_theta'] = self.grad_theta


def _compute_sparse_minimizer(hat_vect_matrix: sparse.csr_matrix, X: sparse.csr_matrix) -> np.ndarray:
    # compute numerator part of minimizer, this will yield a dense vector (size of u or v)
    minimizer = (hat_vect_matrix.multiply(X)).sum(axis=1)
    # divide by norm squared of masked vector
    vector_of_norms = (hat_vect_matrix.multiply(hat_vect_matrix)).sum(axis=1)
    # if some norm is 0 numerator will be 0 too (avoid 0/0)
    vector_of_norms[vector_of_norms < 1e-8] = 1
    minimizer = minimizer / vector_of_norms
    # return np.ndarray representation
    return minimizer.A


def _compute_sparse_gradient(hat_vect_matrix: sparse.csr_matrix, X: sparse.csr_matrix, z: np.ndarray,
                             y: np.ndarray) -> np.ndarray:
    # grad_z = (hat_vect_matrix.multiply(z @ y.T - X)).sum(axis=1) <-- compressed but memory inefficient (`A`=z @ y.T is dense) implementation
    # return grad_z.A
    # sparse matrix are represented by data, rows, cols indices 
    sparse_X_tuple = (X.data, *X.nonzero())
    # create difference matrix sparse representation to avoid passing `hat_vect_matrix` as full dense matrix
    diff_matrix = sparse.csr_matrix((_compute_sparse_difference_matrix(sparse_X_tuple, z, y)), shape=X.shape,
                                    dtype=z.dtype)

    # print("Norm-check", np.linalg.norm(hat_vect_matrix.multiply(diff_matrix).toarray() - hat_vect_matrix.multiply(z @ y.T - X).toarray() ) )

    # sum over rows (axis=1)
    return hat_vect_matrix.multiply(diff_matrix).sum(axis=1)


@njit(parallel=False)
def _compute_sparse_difference_matrix(X_sparse_repr: tuple, z: np.ndarray, y: np.ndarray):
    # compute (uv^T - X) leveraging sparse representation, only compute those elements who aren't zero for X
    X_data, X_rows, X_cols = X_sparse_repr
    data, rows, cols = [], [], []
    for i, (row, col) in enumerate(zip(X_rows, X_cols)):
        entry = (z[row] * y[col] - X_data[i])[0]
        if entry != 0:
            # diff matrix will be 0 (at least) wherever X is 0
            data.append(entry)
            rows.append(row)
            cols.append(col)

    return data, (rows, cols)


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
