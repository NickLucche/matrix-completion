import numpy as np
import os
import csv

from scipy import sparse

class MovieLensDataset:

    def __init__(self, path:str, n_users, n_movies, mode:str='sparse'):
        self.path = path
        self.n_users = n_users
        self.n_movies = n_movies
        # load dataset and build 'augmented' matrix UsersxMovies
        # all args are expected to be >=0
        print(f"Loading MovieLens dataset from {path} with mode {mode}..")
        self.movie_map = {}
        self.movie_counter = 0

        with open(path) as csv_file:
            self.X = self._load_sparse(csv_file) if mode == 'sparse' else self._load_full(csv_file)

    def dataset(self):
        return self.X

    def train_test_split(self, train_perc:int, test_perc:int):
        # TODO: When splitting, you have to make sure movieId and userId both appear in training set
        train_split = int(train_perc * self.X.shape[0] / 100.)
        return self.X[:train_split, :], self.X[train_split:, :]

    def _movie_mapping(self, movie_id: str)->int:
        # estabilishes an enumeration mapping from global movieid defined
        # in the dataset, to a simpler 0-n_movies internal representation
        if movie_id in self.movie_map:
            return self.movie_map[movie_id]
        else:
            self.movie_map[movie_id] = self.movie_counter
            self.movie_counter += 1
            return self.movie_map[movie_id]
    
    def _load_full(self, csv_file)->np.ndarray:
        X = np.zeros((self.n_users, self.n_movies)).astype(np.float64) 
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
            else:
                # enumerate from 0 instead of 1
                user_id = int(row[0])-1
                movie_id = self._movie_mapping(row[1])
                rating = float(row[2])
                X[user_id][movie_id] = rating
            line_count += 1

        print(f'Processed {line_count} lines.')
        print(f"Augmented dataset of size {X.shape} (users x movies) correctly loaded")
        print(f"Dataset contains {np.count_nonzero(X)} ratings ({(line_count-1)/(self.n_movies*self.n_users)}% matrix density)")
        return X

    def _load_sparse(self, csv_file)->sparse.csr_matrix: 
        # construct sparse matrix using rows-cols-data format
        rows = [] # row indices
        cols = [] # cols indices
        data = [] # rating[i] at corresponding rows[i], cols[i] position
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
            else:
                # enumerate from 0 instead of 1
                user_id = int(row[0])-1
                movie_id = self._movie_mapping(row[1])
                rating = float(row[2])

                rows.append(user_id)
                cols.append(movie_id)
                data.append(rating)
            line_count += 1

        print(f'Processed {line_count} lines.')
        print(f"Dataset contains {line_count-1} ratings ({(line_count-1)/(self.n_movies*self.n_users)}% matrix density)")
        # TODO: explain re-scaling trick for float->int
        return sparse.csr_matrix((data, (rows, cols)), shape=(self.n_users, self.n_movies), dtype=np.float64)
        
    # TODO: map movieid to name