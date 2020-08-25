import argparse
from ast import dump

from scipy import sparse
from scipy.sparse import data
from als import ALS, ALSSparse
from dataset import MovieLensDataset
from argparse import ArgumentParser
import numpy as np
import json

def init_vector(shape, normalize=True):
    z = np.abs(np.random.randn(shape)).reshape(-1, 1).astype(np.float64)
    # u /= np.sum(u)
    return z/np.linalg.norm(z) if normalize else z

def average_stats(old_stats, new_run_stats, n):
    for k in new_run_stats:
        if k not in old_stats:
            old_stats[k] = new_run_stats[k]
        else: # running average
            old_stats[k] = 1/n * (new_run_stats[k] + (n-1) * old_stats[k])
    return old_stats

def run_experiment(data: MovieLensDataset, sparse, grad_sensibility=1e-8, num_experiments=10):
    stats = {}
    for i in range(num_experiments):
        u = init_vector(data.n_users, normalize=True)
        v = init_vector(data.n_movies, normalize=True)
        args = [u, v, data.dataset()]
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
    return als

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument('-d', '--dataset-path', help='Absolute path of the csv dataset to load', required=True)
    args = args.parse_args()
    # TODO: make it so we don't need to specify these args
    dataset = MovieLensDataset(args.dataset_path, n_users=610, n_movies=9742, mode='sparse')

    # initialize u, v TODO: INIT FUNDAMENTAL NOTE: `np.abs` TODO: FORGETTING NP.ABS INIT LEADS TO MUCH SLOWER CONVERGENCE (SHOW) 
    
    # for i in range(dataset.dataset().shape[1]):
    #     movie_i_ratings = dataset.dataset()[:, i]
    #     v[i] = movie_i_ratings[movie_i_ratings>0].mean()
    
    # run Alternating Least Squares algorithm
    als = run_experiment(dataset, sparse=True)

    # divide sum of errors on each element by number of elems on which sum is computed
    print("Mean Squared error is:", als.function_eval()/dataset.dataset().getnnz())
    
    # TODO: numba version seems slightly faster in full-mode, test on bigger dataset
    dataset = MovieLensDataset(args.dataset_path, 610, 9742, mode='full')
    
    als = run_experiment(dataset, sparse=False)
    print("Mean Squared error is:", als.function_eval()/np.count_nonzero(dataset.dataset()))

    # TODO: check recommendation
