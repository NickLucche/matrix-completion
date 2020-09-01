import numpy as np
import os
import csv

from scipy import sparse

class MovieLensDataset:

    def __init__(self, path:str, n_users, n_movies, mode:str='sparse'):
        self.path = path
        self.n_users = n_users
        self.n_movies = n_movies
        self.mode = mode
        # load dataset and build 'augmented' matrix UsersxMovies
        # all args are expected to be >=0 (can contain non-integer ratings tho)
        print(f"Loading MovieLens dataset from {path} with mode {mode}..")
        self.movie_map = {}
        self.inverse_movie_map = {}
        self.movie_counter = 0
        self.movie_names = None

        with open(path) as csv_file:
            self.X = self._load_sparse(csv_file) if mode == 'sparse' else self._load_full(csv_file)

    def dataset(self):
        return self.X

    def train_test_split(self, test_size, rnd_seed, min_user_ratings=2, min_movie_ratings=2):
        """ Split dataset into train-test, minding test entries do appear in training
            at least `min_user_ratings` time for each user and `min_movie_ratings` 
            times for each movie. 
            Return generated test set (same shape as original dataset) and train set. 
            Latter is obtained by editing dataset in-place to save memory in case of `full` matrix,
            while a copy is returned in case of `sparse` (changing structure is expensive).
            Original dataset will equal `test + train`.
        Args:
            rnd_seed ([type]): [description]
            test_size: number of ratings test set will have.
            min_user_ratings (int, optional): [description]. Defaults to 2.
            min_movie_ratings (int, optional): [description]. Defaults to 2.
        """
        np.random.seed(rnd_seed)
        # virtually shuffle ratings along axis 0 (only change access pattern)
        perm = np.random.permutation(self.X.shape[0])
        # use sparse format for efficiency
        test_X = sparse.lil_matrix(self.X.shape)
        
        train_X = (self.X.tolil(copy=True) if self.mode == 'sparse' else self.X)

        def num_entries(x):
            if isinstance(x, np.ndarray):
                return np.count_nonzero(x)
            else:
                return x.getnnz()
        def nonzero_indices(x):
            if isinstance(x, np.ndarray):
                x = x.reshape(1, -1)
            _, cols = x.nonzero()
            return cols
        inserted, counter = 0, 0
        while inserted < test_size:
            # go through "shuffled" array, re-start from beginning after n_users iterations
            i = perm[counter % len(perm)]
            counter += 1
            # check whether user i gave more `min_user_ratings`
            if num_entries(train_X[i]) > min_user_ratings:
                # check whether movie j was given more than `min_movie_ratings` ratings
                rated_movies = nonzero_indices(train_X[i])
                for mid in rated_movies:
                    if num_entries(train_X[:, mid]) > min_movie_ratings:
                        # insert rating of movie j by user i into test
                        inserted += 1
                        test_X[i, mid] = train_X[i, mid]

                        # zero-out train set at that position
                        train_X[i, mid] = 0.0
                        break       

        # return train, test
        if self.mode == 'full':
            return self.X, test_X
        elif self.mode == 'sparse':
            return train_X.tocsr(), sparse.csr_matrix(test_X, dtype=np.float64)       

    def _movie_mapping(self, movie_id: str)->int:
        # estabilishes an enumeration mapping from global movieid defined
        # in the dataset, to a simpler 0-n_movies internal representation
        if movie_id in self.movie_map:
            return self.movie_map[movie_id]
        else:
            self.movie_map[movie_id] = self.movie_counter
            self.inverse_movie_map[self.movie_counter] = movie_id
            self.movie_counter += 1
            return self.movie_map[movie_id]
    
    def _load_full(self, csv_file)->np.ndarray:
        # use uint8 to save max space
        X = np.zeros((self.n_users, self.n_movies)).astype(np.uint8) 
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
            else:
                # enumerate from 0 instead of 1
                user_id = int(row[0])-1
                movie_id = self._movie_mapping(row[1])
                # rescale ratings from float to integer range
                rating = MovieLensDataset._rescale_rating(float(row[2]))
                X[user_id][movie_id] = rating
            line_count += 1

        print(f'Processed {line_count} lines.')
        print(f"Augmented dataset of size {X.shape} (users x movies) correctly loaded")
        print(f"Dataset contains {np.count_nonzero(X)} ratings ({(line_count-1)/(self.n_movies*self.n_users)*100}% matrix density)")
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
                # rescale rating from float to int8 range (save space)
                rating = MovieLensDataset._rescale_rating(float(row[2]))

                rows.append(user_id)
                cols.append(movie_id)
                data.append(rating)
            line_count += 1

        print(f'Processed {line_count} lines.')
        print(f"Dataset contains {line_count-1} ratings ({(line_count-1)/(self.n_movies*self.n_users)*100}% matrix density)")
        # TODO: explain re-scaling trick for float->int (save mem by mapping 4.5->5)
        return sparse.csr_matrix((data, (rows, cols)), shape=(self.n_users, self.n_movies), dtype=np.uint8)
        
    # Map internal movieid to actual movie title (retrieved from another file)
    def get_movie_info(self, movie_id):
        # load movies names just-in-time
        if self.movie_names is None:
            self.movie_names = self._load_movies_information('./data/movies.csv')

        # get MovieLens/IMDB movie id
        imdb_movie_id = self.inverse_movie_map[movie_id]
        # get movie information
        return self.movie_names[imdb_movie_id]

    def _load_movies_information(self, path):
        print(f"Loading information about movies from {path}..")
        movie_names = {}
        with open(path, 'r') as f:
            csv_reader = csv.reader(f, delimiter=',')
            for line_count, row in enumerate(csv_reader):
                if line_count > 0:
                   movie_names[row[0]] = {'title': row[1], 'genre':row[2] }
        return movie_names       

    @staticmethod
    def _rescale_rating(rating:float)->int:
        NEW_MAX = 10
        NEW_MIN = 1
        new_range = NEW_MAX - NEW_MIN
        old_range = 5.0 - 0.5
        return (((rating - .5) * new_range) / old_range) + NEW_MIN

    @staticmethod
    def _rescale_back_rating(rating)->float:
        NEW_MAX = 5.0
        NEW_MIN = 0.5
        new_range = NEW_MAX - NEW_MIN
        old_range = 10.0 - 1.
        return (((rating - 1.) * new_range) / old_range) + NEW_MIN

if __name__ == "__main__":
    import time
    m = MovieLensDataset('data/ratings.csv', 610, n_movies=9742, mode='sparse')
    start = time.time()
    train, test = m.train_test_split(1000, 7)
    print("Loading time", time.time()-start)
    print(train.shape, test.shape)
    print(np.sum(train), np.sum(test))
    # print(np.count_nonzero(train), np.count_nonzero(test))
    print(train.getnnz(), test.getnnz())

    m = MovieLensDataset('data/ratings.csv', 610, n_movies=9742, mode='full')
    start = time.time()
    train, test = m.train_test_split(1000, 7)
    print("Loading time", time.time()-start)
    print(train.shape, test.shape)
    print(np.sum(train), np.sum(test))
    print(np.count_nonzero(train), np.count_nonzero(test))