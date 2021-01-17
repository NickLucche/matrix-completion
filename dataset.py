import numpy as np
import os
import csv
from tqdm import tqdm

from scipy import sparse


class MovieLensDataset:
    def __init__(self, path: str, mode: str = 'sparse'):
        self.path = path
        self.n_users = None
        self.n_movies = None
        self.n_ratings = 0
        self.mode = mode
        # load dataset and build 'augmented' matrix UsersxMovies
        # all args are expected to be >=0 (can contain non-integer ratings tho)
        print(f"Loading MovieLens dataset from {path} with mode {mode}..")
        self.movie_map = {}
        self.inverse_movie_map = {}
        self.movie_counter = 0
        self.movie_names = None

        with open(os.path.join(path, 'ratings.csv')) as csv_file:
            self.X = self._load_sparse(
                csv_file) if mode == 'sparse' else self._load_full(csv_file)

    def dataset(self):
        return self.X

    def train_test_split_simple(self, test_size: int):
        """ Naive implementation of what `train_test_split` method does, but leveraging
            prior knowledge on distributions of ratings in dataset: GroupLens ensures that
            `All users selected had rated at least 20 movies` therefore we can sample from 
            dataset without worrying too much.
            Train modified in-place (same shape, zero-out extracted entries), test is returned by copy.
            Mostly useful for quick train-test split on dataset with known properties.
        Args:
            test_size (int): number of ratings test set will have.
        """
        # avoid starting always from first index or you'll get all users with low id
        test_idx = np.random.permutation(self.X.shape[0])
        # get a rnd number of ratings (for some random movie) from each of the selected users
        test_X = sparse.lil_matrix(self.X.shape, dtype=np.uint8) 
        train_X = self.X.tolil() if self.mode == 'sparse' else self.X
        print("Extracting test set..")

        def nonzero_indices(x):
            if isinstance(x, np.ndarray):
                x = x.reshape(1, -1)
            _, cols = x.nonzero()
            return cols

        for i in test_idx:
            if test_size <= 0:
                break
            # compute how many ratings to extract (test_size can be greater than n_users)
            num_ratings = 10  #np.random.randint(10, 15)
            if test_size - num_ratings < 0:
                num_ratings = test_size
            test_size -= num_ratings
            # select only among non-zero ratings
            ratings_to_get = np.random.choice(nonzero_indices(train_X[i]),
                                              size=num_ratings,
                                              replace=False)
            test_X[i, ratings_to_get] = train_X[i, ratings_to_get]
            # zero-out train
            train_X[i, ratings_to_get] = 0

        # get test set ratings
        if self.mode == 'sparse':
            self.X = train_X.tocsr()
            test_X = test_X.tocsr()

        return self.X, test_X

    def train_test_split(self,
                         test_size,
                         n_workers,
                         rnd_seed=7,
                         min_user_ratings=2,
                         min_movie_ratings=2):
        """ Split dataset into train-test, minding test entries do appear in training
            at least `min_user_ratings` time for each user and `min_movie_ratings` 
            times for each movie. 
            Return generated test set (same shape as original dataset) and train set. 
            Latter is obtained by editing dataset in-place to save memory in case of `full` matrix,
            while a copy is returned in case of `sparse` (changing structure is expensive).
            Original dataset will equal `test + train`.
        Args:
            
            test_size: number of ratings test set will have.
            min_user_ratings (int, optional): How many movies must (at least) each user have rated 
            to be inserted in test set. Defaults to 2.
            min_movie_ratings (int, optional): How many ratings must (at least) a movie have received
            to be inserted in test set. Defaults to 2.
        """
        import ray
        from parallel_test_train_split import gen_test_set_task, promise_iterator
        ray.init(ignore_reinit_error=True)

        # use sparse format for efficiency
        test_X = sparse.lil_matrix(self.X.shape, dtype=np.uint8)

        train_X = (self.X.tolil(
            copy=True) if self.mode == 'sparse' else self.X)

        # chunk dim (rows) per worker ~ balanced
        N = [self.n_users // n_workers] * (n_workers - 1)
        N.append(self.n_users - sum(N))  # last worker gets remaining
        print("Rows per worker", N)
        # number of elements to obtain per worker
        M = [test_size // n_workers] * (n_workers - 1)
        M.append(test_size - sum(M))
        print("Elements per worker", M)

        # spawn process and feed tasks
        promises = []
        indices = [0]
        for i, n, m in zip(range(n_workers), N, M):
            trainX_chunk = train_X[indices[i]:indices[i] + n, :]
            promises.append(
                gen_test_set_task.remote(trainX_chunk,
                                         m,
                                         indices[i],
                                         rnd_seed,
                                         min_user_ratings=min_user_ratings,
                                         min_movie_ratings=min_movie_ratings))
            indices.append(indices[i] + n)

        for (test_block, start_idx), n in zip(promise_iterator(promises), N):
            print(f"New job finished from start_idx {start_idx}!")
            print(test_block.shape, start_idx, n)
            test_X[start_idx:start_idx + n, :] = test_block

        # zero-out train set at those position inserted in test set
        train_X[test_X.nonzero()] = 0

        print(
            "NNZ elems train:",
            train_X.getnnz()
            if self.mode == 'sparse' else np.count_nonzero(train_X))
        print("NNZ elems test:", test_X.getnnz())

        # return train, test
        if self.mode == 'full':
            return self.X, test_X
        elif self.mode == 'sparse':
            return train_X.tocsr(), sparse.csr_matrix(test_X, dtype=np.uint8)

    def _movie_mapping(self, movie_id: str) -> int:
        """ Estabilishes an enumeration mapping from global movieid defined
            in the dataset, to a simpler 0-n_movies internal representation
            for direct index access (e.g. X[i, j]=rating user `i` gave to movie `j`).
        Args:
            movie_id (str): original IMDB/Grouplens movie id.
        Returns:
            int: internal id for that movie.
        """
        if movie_id in self.movie_map:
            return self.movie_map[movie_id]
        else:
            self.movie_map[movie_id] = self.movie_counter
            self.inverse_movie_map[self.movie_counter] = movie_id
            self.movie_counter += 1
            return self.movie_map[movie_id]

    def _load_full(self, csv_file) -> np.ndarray:
        """ Method for loading dataset in `full-matrix` mode, without leveraging sparse
            representation but rather standard numpy array operations.
        Args:
            csv_file ([type]): File descriptor pointing to csv-formatted dataset.

        Returns:
            np.ndarray: Dataset in matrix form, where X[i, j] can be read as (rescaled uint8)
            rating user `i` gave to movie `j`.
        """
        # load it as sparse then get dense representation
        X = self._load_sparse(csv_file)
        return X.A

    def _load_sparse(self, csv_file) -> sparse.csr_matrix:
        """ Method for loading dataset in `full-matrix` mode, leveraging sparse
            representation and operations implemented by `scipy.sparse` module.
        Args:
            csv_file ([type]): File descriptor pointing to csv-formatted dataset.

        Returns:
            sparse.csr_matrix: Dataset in compressed row format matrix form, 
            where X[i, j] can be read as (rescaled uint8) rating user `i` gave to movie `j`.
        """
        # construct sparse matrix using rows-cols-data format
        rows = []  # row indices
        cols = []  # cols indices
        data = []  # rating[i] at corresponding rows[i], cols[i] position
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            # expect first row to contain column names
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
            else:
                # enumerate from 0 instead of 1
                user_id = int(row[0]) - 1
                movie_id = self._movie_mapping(row[1])
                # rescale rating from float to int8 range (save space)
                rating = MovieLensDataset._rescale_rating(float(row[2]))

                rows.append(user_id)
                cols.append(movie_id)
                data.append(rating)
            line_count += 1

        self.n_users = len(np.unique(rows))
        self.n_movies = len(np.unique(cols))
        print(
            f'Processed {line_count} lines. {self.n_users} users x {self.n_movies} movies.'
        )
        print(
            f"Dataset contains {line_count-1} ratings ({(line_count-1)/(self.n_movies*self.n_users)*100}% matrix density)"
        )
        self.n_ratings = line_count - 1

        return sparse.csr_matrix((data, (rows, cols)),
                                 shape=(self.n_users, self.n_movies),
                                 dtype=np.uint8)

    def get_movie_info(self, movie_id):
        """ Maps internal movieid to actual movie title (retrieved from another file). """
        # load movies names just-in-time
        if self.movie_names is None:
            self.movie_names = self._load_movies_information(
                os.path.join(self.path, 'movies.csv'))

        # get MovieLens/IMDB movie id
        imdb_movie_id = self.inverse_movie_map[movie_id]
        # get movie information
        return self.movie_names[imdb_movie_id]

    def _load_movies_information(self, path: str):
        """ Reads additional information regarding movies, such as title, year, genre..
            mostly useful for evaluation rather than optimization.
        Args:
            path ([str]): path to movie information file.

        Returns:
            Dictionary mapping name of movie to additional metadata.
        """
        print(f"Loading information about movies from {path}..")
        movie_names = {}
        with open(path, 'r') as f:
            csv_reader = csv.reader(f, delimiter=',')
            for line_count, row in enumerate(csv_reader):
                if line_count > 0:
                    movie_names[row[0]] = {'title': row[1], 'genre': row[2]}
        return movie_names

    @staticmethod
    def _rescale_rating(rating: float) -> int:
        return int(rating * 2)

    @staticmethod
    def _rescale_back_rating(rating) -> float:
        return rating / 2.0


if __name__ == "__main__":
    import time
    m = MovieLensDataset('data/ratings.csv', mode='sparse')
    start = time.time()
    train, test = m.train_test_split(1000, 7)
    print("Loading time", time.time() - start)
    print(train.shape, test.shape)
    print(np.sum(train), np.sum(test))
    # print(np.count_nonzero(train), np.count_nonzero(test))
    print(train.getnnz(), test.getnnz())

    m = MovieLensDataset('data/ratings.csv', 610, n_movies=9742, mode='full')
    start = time.time()
    train, test = m.train_test_split(1000, 7)
    print("Loading time", time.time() - start)
    print(train.shape, test.shape)
    print(np.sum(train), np.sum(test))
    print(np.count_nonzero(train), np.count_nonzero(test))