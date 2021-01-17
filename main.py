from scipy import sparse
from als import ALS, ALSSparse
from dataset import MovieLensDataset
from argparse import ArgumentParser
import numpy as np
import json
import random
import os
import time
from utils import load_matrix, save_matrix
from datetime import datetime


def init_vector(shape, normalize=True):
    # np.random.seed(10)
    z = np.abs(np.random.randn(shape)).reshape(-1, 1).astype(np.float64)
    # u /= np.sum(u)
    return z / np.linalg.norm(z) if normalize else z


def average_stats(old_stats, new_run_stats, n):
    for k in new_run_stats:
        # store all runs list to compute mean/var
        if k == 'fun_evals' or k == 'grad_theta':
            if k not in old_stats:
                old_stats[k] = {str(n): new_run_stats[k]}
            else:
                old_stats[k][str(n)] = new_run_stats[k]
        elif k not in old_stats:
            old_stats[k] = new_run_stats[k]
        else:  # running average
            old_stats[k] = 1 / n * (new_run_stats[k] + (n - 1) * old_stats[k])
    return old_stats


def run_experiment(data: MovieLensDataset,
                   sparse=True,
                   grad_sensibility=1e-8,
                   num_experiments=1,
                   warmup=0,
                   workers=8):
    date = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    # try to load matrices first
    try:
        print("Loading train and test split from /tmp/..")
        trainX = load_matrix(f'trainX_{"sparse" if sparse else "full"}',
                             sparse)
        testX = load_matrix(f'testX_{"sparse" if sparse else "full"}', sparse)
    except:
        print("Loading failed, generating train-test split now..")
        # %5 test size
        test_set_size = data.n_ratings // 20
        # trainX, testX = data.train_test_split(test_set_size, workers)
        trainX, testX = data.train_test_split_simple(test_set_size)
        print(f"Saving train and test set to /tmp/ first..")
        save_matrix(f'trainX_{"sparse" if sparse else "full"}', trainX)
        save_matrix(f'testX_{"sparse" if sparse else "full"}', testX)

    # print(trainX.shape, testX.shape)
    # optional warmup
    for _ in range(warmup):
        u = init_vector(data.n_users, normalize=True)
        v = init_vector(data.n_movies, normalize=True)
        args = [u, v, trainX]
        als = ALSSparse(*args) if sparse else ALS(*args)
        u, v = als.fit(eps_g=grad_sensibility)

    stats = {}
    start = time.time()
    for i in range(num_experiments):
        u = init_vector(data.n_users, normalize=True)
        v = init_vector(data.n_movies, normalize=True)
        args = [u, v, trainX]
        als = ALSSparse(*args) if sparse else ALS(*args)
        # run Alternating Least Squares algorithm
        u, v = als.fit(eps_g=grad_sensibility)
        # average results
        stats = average_stats(stats, als.stats, i + 1)
    end = time.time()
    # additional context info non depending from experiment results
    stats['number_of_ratings'] = trainX.getnnz(
    ) if sparse else np.count_nonzero(trainX)
    stats['dataset_path'] = data.path
    stats['grad_sensibility'] = grad_sensibility
    stats['theta_diff_sensibility'] = 1e-10
    stats['num_experiments'] = num_experiments
    stats['warmup_cycles'] = warmup
    stats['experiments_total_runtime'] = end - start
    stats['date'] = date
    stats['train_mse'] = als.function_eval() / stats['number_of_ratings']
    print("Train Mean Squared error is:", stats['train_mse'])

    # free memory before testing
    del trainX
    del data

    # test on test set
    test_mse = evaluate(als.u, als.v, testX, "sparse" if sparse else "full")

    stats['test_mse'] = test_mse
    # save results
    print("Saving results..")
    with open(f'data/als_{"sparse" if sparse else "full"}_{date}.json',
              'w') as f:
        json.dump(stats, f, indent=4)

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
        rating = d.dataset()[userx,
                             m_id] if m_id in movies_already_rated else float(
                                 als.u[userx] * als.v[m_id])
        movie_ratings[str(mdbid)] = {'title': mdbid, 'rating': f'{rating:.2f}'}
    print(json.dumps(movie_ratings, sort_keys=False, indent=4))


def evaluate(u: np.ndarray, v: np.ndarray, test_X, mode: str):
    print(u.shape, v.shape, test_X.shape, mode)
    a = ALSSparse(u, v, test_X) if mode == 'sparse' else ALS(
        u, v, test_X.toarray())
    mse = a.function_eval() / test_X.getnnz()
    print("Test set MSE:", mse)
    return mse
    # M = sparse.csr_matrix(test_X, dtype=np.bool)
    # print("Test set MSE:", ((M.multiply(u @ v.T)/2.) - (test_X.astype(np.float32)/2.)).power(2).sum() )


# TODO: setup for experiments
if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument('-d',
                      '--dataset-path',
                      help='Absolute path of the csv dataset to load',
                      required=True)
    args.add_argument('-s',
                      '--save-path',
                      help='Directory where to save factorization results to',
                      default='./data/')
    args.add_argument(
        '-e',
        '--n-experiments',
        help='Number of experiments/runs to perform for each mode',
        type=int,
        default=1,
        required=False)
    args.add_argument('--warmup',
                      help='Number of warmup runs to perform for each mode',
                      type=int,
                      default=0,
                      required=False)
    args.add_argument(
        '-g',
        '--grad-sensibility',
        help='Sensibility/eps of the gradient in the search for a solution',
        type=float,
        default=1e-8,
        required=False)
    args.add_argument(
        '-w',
        '--n-workers',
        help='Number of workers used to split dataset into test-train',
        type=int,
        default=8)
    args.add_argument('-v',
                      '--verbose',
                      help='Show some recommendations and additional output',
                      default=False,
                      action='store_true')
    args = args.parse_args()

    dataset = MovieLensDataset(args.dataset_path, mode='sparse')

    # another init method
    # for i in range(dataset.dataset().shape[1]):
    #     movie_i_ratings = dataset.dataset()[:, i]
    #     v[i] = movie_i_ratings[movie_i_ratings>0].mean()

    # run Alternating Least Squares algorithm
    als = run_experiment(dataset,
                         sparse=True,
                         grad_sensibility=args.grad_sensibility,
                         num_experiments=args.n_experiments,
                         warmup=args.warmup,
                         workers=args.n_workers)

    # divide sum of errors on each element by number of elems on which sum is computed

    # show some recommendations (optional)
    if args.verbose:
        userx = random.randint(0, dataset.n_users)
        print(
            f"Showing some of the proposed recommendation for user {userx}..")
        show_movie_recommendations(dataset)
        print(f"Storing vectors u, v to disk {args.save_path}..")

    # store latest feature vectors
    np.save(os.path.join(args.save_path, 'sparse_U.npy'), als.u)
    np.save(os.path.join(args.save_path, 'sparse_V.npy'), als.v)

    # dense mode
    dataset = MovieLensDataset(args.dataset_path, mode='full')

    als = run_experiment(dataset,
                         sparse=False,
                         grad_sensibility=args.grad_sensibility,
                         num_experiments=args.n_experiments,
                         warmup=args.warmup,
                         workers=args.n_workers)
    # print("Mean Squared error is:",
    #       als.function_eval() / np.count_nonzero(dataset.dataset()))

    print(f"Storing vectors u, v to disk {args.save_path}..")
    np.save(os.path.join(args.save_path, 'full_U.npy'), als.u)
    np.save(os.path.join(args.save_path, 'full_V.npy'), als.v)