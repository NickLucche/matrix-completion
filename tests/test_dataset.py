import unittest
import sys, os
sys.path.append(os.path.curdir)
from dataset import MovieLensDataset
import numpy as np


class DatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset_sparse = MovieLensDataset('./data',
                                               n_users=610,
                                               n_movies=9742,
                                               mode='sparse')
        self.dataset_full = MovieLensDataset('./data',
                                             n_users=610,
                                             n_movies=9742,
                                             mode='full')

    def tearDown(self) -> None:
        return super().tearDown()

    def test_simple_train_test_split_full(self):
        test_size = self.dataset_full.n_ratings // 20
        train, test = self.dataset_full.train_test_split_simple(test_size)
        self.assertEqual(np.count_nonzero(test), test_size)
        self.assertEqual(test.shape, train.shape)

    def test_simple_train_test_split_sparse(self):
        test_size = self.dataset_sparse.n_ratings // 20
        train, test = self.dataset_sparse.train_test_split_simple(test_size)
        self.assertEqual(test.getnnz(), test_size)
        self.assertEqual(test.shape, train.shape)
