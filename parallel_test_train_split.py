import ray
import numpy as np
from scipy import sparse
import time

@ray.remote
def gen_test_set_task(matrix_block: sparse.lil_matrix, n_elements, start_idx, rnd_seed, min_user_ratings=2, min_movie_ratings=2):
    def num_entries(x):
            if isinstance(x, np.ndarray):
                return np.count_nonzero(x)
            else:
                return x.getnnz()
    def nonzero_indices(x): # takes about ~0.2ms
        if isinstance(x, np.ndarray):
            x = x.reshape(1, -1)
        _, cols = x.nonzero()
        return cols
    # access array in random order (virtual shuffling, create different test sets)
    np.random.seed(rnd_seed)
    perm = np.random.permutation(matrix_block.shape[0])
    inserted = 0
    counter = 0

    test_X = sparse.lil_matrix(matrix_block.shape, dtype=np.uint8)
    start = time.time()
    while inserted < n_elements:
        if counter>0 and counter % 10000 == 0:
            print(inserted,'/', n_elements)
            now = time.time()
            print(f"It took {now-start}s")
            start = now
        # go through "shuffled" array, re-start from beginning after n_users iterations
        row_idx = perm[counter % len(perm)]
        counter += 1
        # check whether user i gave more ratings than `min_user_ratings`
        rated_movies = nonzero_indices(matrix_block[row_idx, :])
        if len(rated_movies) > min_user_ratings:            
            # avoid starting always from first index or you'll get all movies with low id
            for k in np.random.permutation(rated_movies.shape[0]):
                mid = rated_movies[k]
                # check whether movie j was given more than `min_movie_ratings` ratings
                if num_entries(matrix_block[:, mid]) > min_movie_ratings:
                    # insert rating of movie j by user i into test
                    inserted += 1
                    test_X[row_idx, mid] = matrix_block[row_idx, mid]

                    # zero-out train set at that position
                    # matrix_block[row_idx, mid] = 0
                    break     
    print("Inserted:", inserted)
    return test_X, start_idx

def promise_iterator(promises, timeout=1.0):
    # start_time = time.time()
    total_work = len(promises)
    done_work = 0
    while len(promises):
        available, promises = ray.wait(promises, len(promises), timeout)
        new_results = ray.get(available)

        for x in new_results:
            yield x

        done_work += len(new_results)
        # print(f"{time.time() - start_time}  {done_work}/{total_work}")

if __name__ == "__main__":
    ray.init()
    # sparse random doesn't support uint8 generation
    # X = sparse.random(16200, 6200, density=0.25, format='lil')
    # X = X.astype(np.uint8)
    # X = sparse.lil_matrix([[dc(X), dc(X)], [dc(X), dc(X)]])
    from dataset import MovieLensDataset
    data = MovieLensDataset('data/', 610, 9742, 'full')
    X = data.dataset()
    testX = sparse.lil_matrix(X.shape, dtype=np.uint8)
    # print(np.sum(X.toarray()))
    print(np.sum(X))

    n = 100
    promises = [gen_test_set_task.remote(X[i*n:(i+1)*n, :], 100, i*n) for i in range(4)]
    for test_block, start_idx in promise_iterator(promises):
        print(f"New job finished from start_idx {start_idx}!")
        testX[start_idx:start_idx+n, :] = test_block
        print(np.sum(testX.toarray()))
    # zero-out train set at those position inserted in test set
    X[testX.nonzero()] = 0


    print(np.sum(X))
    print(np.sum(testX.toarray()))