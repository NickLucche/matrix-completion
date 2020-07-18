import numpy as np
import os
import csv

class MovieLensDataset:

    def __init__(self, path:str, n_users, n_movies):
        self.path = path
        # load dataset and build 'augmented' matrix UsersxMovies
        # all args are expected to be >=0
        self.X = np.zeros((n_users, n_movies)).astype(np.uint8) 
        print(f"Loading MovieLens dataset from {path}..")
        self.movie_map = {}
        self.movie_counter = 0

        with open(path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    print(f'Column names are {", ".join(row)}')
                else:
                    # enumerate from 0 instead of 1
                    user_id = int(row[0])-1
                    movie_id = self._movie_mapping(row[1])
                    rating = int(float(row[2]))
                    self.X[user_id][movie_id] = rating
                line_count += 1
            print(f'Processed {line_count} lines.')

        print(f"Augmented dataset of size {self.X.shape} (users x movies) correctly loaded")
        print(f"Dataset contains {np.count_nonzero(self.X)} ratings ({(line_count-1)/(n_movies*n_users)}%)")

    def dataset(self):
        return self.X

    def train_test_split(self, train_perc:int, test_perc:int):
        train_split = int(train_perc * len(self.X) / 100.)
        return self.X[:train_split, :], self.X[train_split:, :]

    def _movie_mapping(self, movie_id: str)->int:
        # estabilishes an enumeration mapping from global movieid defined
        # in the dataset, to a simpler 0-n_movies internal representation
        if movie_id in self.movie_map:
            return self.movie_map[movie_id]
        else:
            self.movie_map[movie_id] = self.movie_counter
            self.movie_counter += 1

    # TODO: map movieid to name