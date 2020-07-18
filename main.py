import argparse
from als import ALS
from dataset import MovieLensDataset
from argparse import ArgumentParser
import numpy as np

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument('-d', '--dataset-path', help='Absolute path of the csv dataset to load', required=True)
    args = args.parse_args()
    # TODO: make it so we don't need to specify these args
    dataset = MovieLensDataset(args.dataset_path, 610, 9742)
    train, test = dataset.train_test_split(70, 30)

    # initialize u, v todo: other kind of init
    u = np.random.randn(610).reshape(-1, 1).astype(np.float64)
    v = np.random.randn(9742).reshape(-1, 1).astype(np.float64)

    # run Alternating Least Squares algorithm
    als = ALS(u, v, train)
    u, v = als.fit()

    # test u, v on test set
    test_M = (np.ones_like(test) * test).astype(np.bool)
    print("Squared error on test set is:", np.sum((u @ v.T * test_M - test)**2))

    # TODO: check recommendation
