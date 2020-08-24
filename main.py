import argparse

from scipy import sparse
from scipy.sparse import data
from als import ALS, ALSSparse
from dataset import MovieLensDataset
from argparse import ArgumentParser
import numpy as np
import time

def positive_init(shape, data):
    return np.random.uniform(data.min(), data.max(), shape)

def run_experiment():
    pass

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument('-d', '--dataset-path', help='Absolute path of the csv dataset to load', required=True)
    args = args.parse_args()
    # TODO: make it so we don't need to specify these args
    sparse_dataset = MovieLensDataset(args.dataset_path, n_users=610, n_movies=9742, mode='sparse')

    # initialize u, v TODO: INIT FUNDAMENTAL NOTE: `np.abs` TODO: FORGETTING NP.ABS INIT LEADS TO MUCH SLOWER CONVERGENCE (SHOW) 
    u = np.abs(np.random.randn(610)).reshape(-1, 1).astype(np.float64)
    # u /= np.sum(u)
    u /= np.linalg.norm(u)
    # u = positive_init(610, dataset.dataset()).reshape(-1, 1).astype(np.float64)

    # v = positive_init(9742, dataset.dataset()).reshape(-1, 1).astype(np.float64)
    v = np.abs(np.random.randn(9742)).reshape(-1, 1).astype(np.float64)
    # v /= np.sum(v)
    v /= np.linalg.norm(v)
    # v = np.random.randn(9742).reshape(-1, 1).astype(np.float64)
    # for i in range(dataset.dataset().shape[1]):
    #     movie_i_ratings = dataset.dataset()[:, i]
    #     v[i] = movie_i_ratings[movie_i_ratings>0].mean()
    
    # run Alternating Least Squares algorithm
    als = ALSSparse(u, v, sparse_dataset.dataset())
    u, v = als.fit(max_iter=100)#, eps_g=1e-8)

    # divide sum of errors on each element by number of elems on which sum is computed
    print("Mean Squared error is:", als.function_eval()/sparse_dataset.dataset().getnnz())

    # test u, v on test set
    # test_M = (np.ones_like(test) * test).astype(np.bool)
    # test_M = sparse.csr_matrix(test, dtype=np.bool)
    # print("Mean Squared error on test set is:", (test_M.multiply(u @ v.T) - test).power(2).sum()/test_M.shape[0])
    # print(u)
    # print(v)
    
    # TODO: numba version seems slightly faster in full-mode, test on bigger dataset
    dataset = MovieLensDataset(args.dataset_path, 610, 9742, mode='full')
    print("Dataset diff:", np.linalg.norm(sparse_dataset.dataset()-dataset.dataset()))
    # initialize u, v todo: other kind of init
    u = np.abs(np.random.randn(610)).reshape(-1, 1).astype(np.float64)
    u /= np.linalg.norm(u)
    v = np.abs(np.random.randn(9742)).reshape(-1, 1).astype(np.float64)
    v /= np.linalg.norm(v)
    
    # run Alternating Least Squares algorithm
    als = ALS(u, v, dataset.dataset())
    u, v = als.fit()

    # TODO: sparse test
    # test u, v on test set
    print("Mean Squared error is:", als.function_eval()/np.count_nonzero(dataset.dataset()))

    # TODO: check recommendation
