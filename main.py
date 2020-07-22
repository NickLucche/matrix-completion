import argparse

from scipy import sparse
from scipy.sparse import data
from als import ALS, ALSSparse
from dataset import MovieLensDataset
from argparse import ArgumentParser
import numpy as np

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument('-d', '--dataset-path', help='Absolute path of the csv dataset to load', required=True)
    args = args.parse_args()
    # TODO: make it so we don't need to specify these args
    dataset = MovieLensDataset(args.dataset_path, 610, 9742, mode='sparse')
    train, test = dataset.train_test_split(70, 30)
    # print(train)

    # initialize u, v TODO: explain init
    u = np.abs(np.random.randn(610)).reshape(-1, 1).astype(np.float64)
    u /= np.sum(u)
    v = np.abs(np.random.randn(9742)).reshape(-1, 1).astype(np.float64)
    v /= np.sum(v)
    # v = np.random.randn(9742).reshape(-1, 1).astype(np.float64)
    # for i in range(dataset.dataset().shape[1]):
    #     movie_i_ratings = dataset.dataset()[:, i]
    #     v[i] = movie_i_ratings[movie_i_ratings>0].mean()
    
    # run Alternating Least Squares algorithm
    als = ALSSparse(u, v, dataset.dataset())
    u, v = als.fit(max_iter=100)

    print("Mean Squared error on test set is:", als.function_eval()/dataset.dataset().shape[0])

    # test u, v on test set
    # test_M = (np.ones_like(test) * test).astype(np.bool)
    # test_M = sparse.csr_matrix(test, dtype=np.bool)
    # print("Mean Squared error on test set is:", (test_M.multiply(u @ v.T) - test).power(2).sum()/test_M.shape[0])
    # print(u)
    # print(v)


    # dataset = MovieLensDataset(args.dataset_path, 610, 9742, mode='full')
    # train, test = dataset.train_test_split(70, 30)

    # # initialize u, v todo: other kind of init
    # u = np.random.randn(610).reshape(-1, 1).astype(np.float64)
    # v = np.random.randn(9742).reshape(-1, 1).astype(np.float64)
    # print(type(train))
    # # run Alternating Least Squares algorithm
    # als = ALS(u, v, train)
    # u, v = als.fit()

    # # TODO: sparse test
    # # test u, v on test set
    # test_M = (np.ones_like(test) * test).astype(np.bool)
    # # test_M = sparse.csr_matrix(test, dtype=np.bool)
    # # print("Squared error on test set is:", (test_M.multiply(u @ v.T) - test).power(2).sum())
    # print("Squared error on test set is:", np.sum((test_M * (u @ v.T) - test) ** 2))

    # TODO: check recommendation
