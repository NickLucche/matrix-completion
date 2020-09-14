from scipy import sparse
from als import ALS, ALSSparse
from dataset import MovieLensDataset
from argparse import ArgumentParser
import numpy as np
import json
import random
import os
import scipy
from utils import load_matrix, save_matrix

def init_vector(shape, normalize=True):
    # np.random.seed(10)
    z = np.abs(np.random.randn(shape)).reshape(-1, 1).astype(np.float64)
    # u /= np.sum(u)
    return z/np.linalg.norm(z) if normalize else z
# TODO: store test-set results, store gradients (grad theta) at each iteration, test parallelization
def average_stats(old_stats, new_run_stats, n):
    for k in new_run_stats:
        # keep latest run list
        if k == 'fun_evals' or k == 'grad_theta':
            old_stats[k] = new_run_stats[k]
        elif k not in old_stats:
            old_stats[k] = new_run_stats[k]
        else: # running average
            old_stats[k] = 1/n * (new_run_stats[k] + (n-1) * old_stats[k])
    return old_stats

def run_experiment(data: MovieLensDataset, sparse=True, grad_sensibility=1e-8, num_experiments=1, warmup=0):
    # try to load matrices first
    try:
        print("Loading train and test split from /tmp/..")
        trainX = load_matrix(f'trainX_{"sparse" if sparse else "full"}', sparse)
        testX = load_matrix(f'testX_{"sparse" if sparse else "full"}', sparse)
    except:
        print("Loading failed, generating train-test split now..")
        # %5 test size
        test_set_size = data.n_ratings//20
        trainX, testX = data.train_test_split(test_set_size, 7)
        print(f"Saving train and test set to /tmp/ first..")
        save_matrix(f'trainX_{"sparse" if sparse else "full"}', trainX)
        save_matrix(f'testX_{"sparse" if sparse else "full"}', testX)
        
    print(trainX.shape, testX.shape)
    # optional warmup
    for _ in range(warmup):
        u = init_vector(data.n_users, normalize=True)
        v = init_vector(data.n_movies, normalize=True)
        args = [u, v, trainX]
        als = ALSSparse(*args) if sparse else ALS(*args)
        u, v = als.fit(eps_g=grad_sensibility)

    stats = {}
    for i in range(num_experiments):
        u = init_vector(data.n_users, normalize=True)
        v = init_vector(data.n_movies, normalize=True)
        args = [u, v, trainX]
        als = ALSSparse(*args) if sparse else ALS(*args)
        # run Alternating Least Squares algorithm
        u, v = als.fit(eps_g=grad_sensibility)
        # average results
        stats = average_stats(stats, als.stats, i+1)
    # save results
    print("Saving results..")
    print(json.dumps(stats, sort_keys=True, indent=4))
    with open(f'data/als_{"sparse" if sparse else "full"}_{num_experiments}_runs.json', 'w') as f:
        json.dump(stats, f)

    # free memory before testing
    del trainX
    del data

    # test on test set
    evaluate(als.u, als.v, testX, "sparse" if sparse else "full")
    return als

def show_movie_recommendations(d: MovieLensDataset):
    # sample k random movies already rated by user x in dataset
    movies_already_rated = random.sample(list(d.dataset()[userx].indices), k=5)
    # print(userx, movies_already_rated)
    # sample k random movies among all possible (use movie_counter since some movies might have multiple ratings)
    movie_list = random.sample(range(d.movie_counter), k=5)
    movie_ratings = {}
    # format result
    for m_id in movies_already_rated + movie_list:
        mdbid = d.get_movie_info(m_id)
        # get original (re-mapped) rating if present else compute it from factorization
        rating = d.dataset()[userx, m_id] if m_id in movies_already_rated else float(als.u[userx] * als.v[m_id])
        movie_ratings[str(mdbid)] = {'title': mdbid,
         'rating': f'{rating:.2f}'}
    print(json.dumps(movie_ratings, sort_keys=False, indent=4))

def evaluate(u:np.ndarray, v:np.ndarray, test_X, mode:str):
    print(u.shape, v.shape, test_X.shape, mode)
    # compute mask from M
    a = ALSSparse(u, v, test_X) if mode=='sparse' else ALS(u, v, test_X.toarray())
    print("Test set MSE:", a.function_eval()/ test_X.getnnz() )
    
    # M = sparse.csr_matrix(test_X, dtype=np.bool)
    # print("Test set MSE:", ((M.multiply(u @ v.T)/2.) - (test_X.astype(np.float32)/2.)).power(2).sum() )
    


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument('-d', '--dataset-path', help='Absolute path of the csv dataset to load', required=True)
    args.add_argument('-s', '--save-path', help='Directory where to save factorization results to', default='./data/')
    args.add_argument('-u', '--n-users', help='Number of users present in the dataset', type=int, required=True)
    args.add_argument('-m', '--n-movies', help='Number of movies present in the dataset', type=int, required=True)
    args.add_argument('-w', '--n-workers', help='Number of workers used to split dataset into test-train', type=int, default=8)
    args = args.parse_args()
    # TODO: we could use float32 if not for numba
    dataset = MovieLensDataset(args.dataset_path, n_users=args.n_users, n_movies=args.n_movies, mode='sparse')

    # initialize u, v TODO: INIT FUNDAMENTAL NOTE: `np.abs` TODO: FORGETTING NP.ABS INIT LEADS TO MUCH SLOWER CONVERGENCE (SHOW) 
    
    # for i in range(dataset.dataset().shape[1]):
    #     movie_i_ratings = dataset.dataset()[:, i]
    #     v[i] = movie_i_ratings[movie_i_ratings>0].mean()
    
    # run Alternating Least Squares algorithm
    als = run_experiment(dataset, sparse=True)

    # divide sum of errors on each element by number of elems on which sum is computed
    print("Mean Squared error is:", als.function_eval()/dataset.dataset().getnnz())
    
    # show some recommendations
    userx = random.randint(0, dataset.n_users)
    print(f"Showing some of the proposed recommendation for user {userx}..")
    show_movie_recommendations(dataset)
    print(f"Storing vectors u, v to disk {args.save_path}..")
    np.save(os.path.join(args.save_path, 'sparse_U.npy'), als.u)
    np.save(os.path.join(args.save_path, 'sparse_V.npy'), als.v)
    

    # TODO: numba version seems slightly faster in full-mode, test on bigger dataset
    dataset = MovieLensDataset(args.dataset_path, n_users=args.n_users, n_movies=args.n_movies, mode='full')
    
    als = run_experiment(dataset, sparse=False)
    print("Mean Squared error is:", als.function_eval()/np.count_nonzero(dataset.dataset()))

    print(f"Storing vectors u, v to disk {args.save_path}..")
    np.save(os.path.join(args.save_path, 'full_U.npy'), als.u)
    np.save(os.path.join(args.save_path, 'full_V.npy'), als.v)