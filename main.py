from scipy import sparse
from scipy.sparse import data
from als import ALS, ALSSparse
from dataset import MovieLensDataset
from argparse import ArgumentParser
import numpy as np
import json
import random
import os

def init_vector(shape, normalize=True):
    z = np.abs(np.random.randn(shape)).reshape(-1, 1).astype(np.float64)
    # u /= np.sum(u)
    return z/np.linalg.norm(z) if normalize else z

def average_stats(old_stats, new_run_stats, n):
    for k in new_run_stats:
        # keep latest function evaluation series
        if k == 'fun_evals':
            old_stats[k] = new_run_stats[k]
        if k not in old_stats:
            old_stats[k] = new_run_stats[k]
        else: # running average
            old_stats[k] = 1/n * (new_run_stats[k] + (n-1) * old_stats[k])
    return old_stats

def run_experiment(data: MovieLensDataset, sparse, grad_sensibility=1e-8, num_experiments=10, warmup=2):
    # optional warmup
    for _ in range(warmup):
        u = init_vector(data.n_users, normalize=True)
        v = init_vector(data.n_movies, normalize=True)
        args = [u, v, data.dataset()]
        als = ALSSparse(*args) if sparse else ALS(*args)
        u, v = als.fit(eps_g=grad_sensibility)

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

def show_movie_recommendations(d: MovieLensDataset):
    # sample k random movies already rated by user x in dataset
    movies_already_rated = random.sample(list(d.dataset()[userx].indices), k=5)
    print(userx, movies_already_rated)
    # sample k random movies among all possible (use movie_counter since some movies might have multiple ratings)
    movie_list = random.sample(range(d.movie_counter), k=5)
    movie_ratings = {}
    # format result
    for m_id in movies_already_rated + movie_list:
        mdbid = d.get_movie_info(m_id)
        # get original rating if present else compute it from factorization
        rating = d.dataset()[userx, m_id] if m_id in movies_already_rated else float(als.u[userx] * als.v[m_id])
        movie_ratings[str(mdbid)] = {'title': mdbid,
         'rating': f'{rating:.2f}'}
    print(json.dumps(movie_ratings, sort_keys=False, indent=4))


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument('-d', '--dataset-path', help='Absolute path of the csv dataset to load', required=True)
    args.add_argument('-s', '--save-path', help='Directory where to save factorization results to', default='./data/')
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
    
    # show some recommendations
    userx = random.randint(0, dataset.n_users)
    print(f"Showing some of the proposed recommendation for user {userx}..")
    show_movie_recommendations(dataset)
    print(f"Storing vectors u, v to disk {args.save_path}..")
    np.save(os.path.join(args.save_path, 'sparse_U.npy'), als.u)
    np.save(os.path.join(args.save_path, 'sparse_V.npy'), als.v)


    # TODO: numba version seems slightly faster in full-mode, test on bigger dataset
    dataset = MovieLensDataset(args.dataset_path, 610, 9742, mode='full')
    
    als = run_experiment(dataset, sparse=False)
    print("Mean Squared error is:", als.function_eval()/np.count_nonzero(dataset.dataset()))

    print(f"Storing vectors u, v to disk {args.save_path}..")
    np.save(os.path.join(args.save_path, 'full_U.npy'), als.u)
    np.save(os.path.join(args.save_path, 'full_V.npy'), als.v)