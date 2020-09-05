import numpy as np
import os
import csv
import ray
from parallel_test_train_split import gen_test_set_task, promise_iterator

from scipy import sparse

class MovieLensDataset:

    def __init__(self, path:str, n_users, n_movies, mode:str='sparse'):
        self.path = path
        self.n_users = n_users
        self.n_movies = n_movies
        self.n_ratings = 0
        self.mode = mode
        # load dataset and build 'augmented' matrix UsersxMovies
        # all args are expected to be >=0 (can contain non-integer ratings tho)
        print(f"Loading MovieLens dataset from {path} with mode {mode}..")
        self.movie_map = {}
        self.inverse_movie_map = {}
        self.movie_counter = 0
        self.movie_names = None

        with open(os.path.join(path,'ratings.csv')) as csv_file:
            self.X = self._load_sparse(csv_file) if mode == 'sparse' else self._load_full(csv_file)

    def dataset(self):
        return self.X

    def train_test_split(self, test_size, n_workers, rnd_seed=7, min_user_ratings=2, min_movie_ratings=2):
        """ Split dataset into train-test, minding test entries do appear in training
            at least `min_user_ratings` time for each user and `min_movie_ratings` 
            times for each movie. 
            Return generated test set (same shape as original dataset) and train set. 
            Latter is obtained by editing dataset in-place to save memory in case of `full` matrix,
            while a copy is returned in case of `sparse` (changing structure is expensive).
            Original dataset will equal `test + train`.
        Args:
            
            test_size: number of ratings test set will have.
            min_user_ratings (int, optional): [description]. Defaults to 2.
            min_movie_ratings (int, optional): [description]. Defaults to 2.
        """
        ray.init(ignore_reinit_error=True)
        
        # use sparse format for efficiency
        test_X = sparse.lil_matrix(self.X.shape, dtype=np.uint8)
        
        train_X = (self.X.tolil(copy=True) if self.mode == 'sparse' else self.X)

        # chunk dim (rows) per worker ~ balanced
        N = [self.n_users // n_workers] * (n_workers-1)
        N.append(self.n_users-sum(N))  # last worker gets remaining
        print("Rows per worker", N)
        # number of elements to obtain per worker
        M = [test_size // n_workers] * (n_workers-1)
        M.append(test_size-sum(M))
        print("Elements per worker", M)

        # spawn process and feed tasks
        promises = []
        indices = [0]
        for i, n, m in zip(range(n_workers), N, M):
            trainX_chunk = train_X[indices[i]:indices[i]+n, :]
            promises.append(gen_test_set_task.remote(trainX_chunk, m, indices[i], rnd_seed))
            indices.append(indices[i] + n)

        for (test_block, start_idx), n in zip(promise_iterator(promises), N):
            print(f"New job finished from start_idx {start_idx}!")
            print(test_block.shape, start_idx, n)
            test_X[start_idx:start_idx+n, :] = test_block

        # zero-out train set at those position inserted in test set
        train_X[test_X.nonzero()] = 0

        print("NNZ elems train:",train_X.getnnz() if self.mode=='sparse' else np.count_nonzero(train_X))
        print("NNZ elems test:", test_X.getnnz())

        # return train, test
        if self.mode == 'full':
            return self.X, test_X
        elif self.mode == 'sparse':
            return train_X.tocsr(), sparse.csr_matrix(test_X, dtype=np.uint8)       

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
        self.n_ratings = line_count-1
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
        self.n_ratings = line_count-1
        # TODO: explain re-scaling trick for float->int (save mem by mapping 4.5->5) leads to slower convergence and higher ts error (local)
        return sparse.csr_matrix((data, (rows, cols)), shape=(self.n_users, self.n_movies), dtype=np.uint8)
        
    # Map internal movieid to actual movie title (retrieved from another file)
    def get_movie_info(self, movie_id):
        # load movies names just-in-time
        if self.movie_names is None:
            self.movie_names = self._load_movies_information(os.path.join(self.path,'movies.csv'))

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
        return int(rating * 2)

    @staticmethod
    def _rescale_back_rating(rating)->float:
        return rating / 2.0

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