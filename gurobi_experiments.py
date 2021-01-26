import numpy as np
from als import ALSSparse
import gurobipy as grb
import time

from dataset import MovieLensDataset
from utils import load_matrix, save_matrix, sparse_matrix_to_csv
from argparse import ArgumentParser
from main import evaluate

MAX_ITER = 100
SEED = 10

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument('-d',
                      '--dataset-path',
                      help='Absolute path of the csv dataset to load',
                      required=False)
    args.add_argument(
        '-t',
        '--time-limit',
        help='How long do you want Gurobi to run for (in seconds)',
        type=int,
        required=False)
    args.add_argument('-lb',
                      '--lower-bound',
                      help='Vector u,v lower bound constraint',
                      type=int,
                      default=-20.0,
                      required=False)
    args.add_argument('-ub',
                      '--upper-bound',
                      help='Vector u,v upper bound constraint',
                      type=int,
                      default=20.0,
                      required=False)
    args.add_argument(
        '-nn',
        '--non-negative',
        help=
        'Setting this will solve least-squares with nonnegativity constraints',
        action='store_true',
        default=False)
    args.add_argument(
        '-w',
        '--n-workers',
        help='Number of workers used to split dataset into test-train',
        type=int,
        default=8)
    args = args.parse_args()

    dataset = MovieLensDataset(args.dataset_path, mode='sparse')
    try:
        print("Loading train and test split from /tmp/..")
        trainX = load_matrix(f'trainX_sparse', True)
        testX = load_matrix(f'testX_sparse', True)
    except:
        print("Loading failed, generating train-test split now..")
        # %5 test size
        test_set_size = dataset.n_ratings // 20
        # trainX, testX = dataset.train_test_split_simple(test_set_size)
        trainX, testX = dataset.train_test_split(test_set_size, args.n_workers)
        print(f"Saving train and test set to /tmp/ first..")
        save_matrix(f'trainX_sparse', trainX)
        save_matrix(f'testX_sparse', testX)

    # train matrix to csv
    # TODO: trainX retains original shape, dumped dataset does not (it shrinks on reload)
    # n_users and movies (shape) must be kept constant (make sure each movie has at least 2 ratings..)
    
    print("Storing train matrix to csv restoring to original format..")
    # dataset.to_csv('/tmp/ratings.csv', trainX)
    # dataset = MovieLensDataset('/tmp/', mode='sparse')

    # setup gurobi opt model
    opt_model = grb.Model(name="ALS Compare Model")
    # needed to specify non convex problem
    opt_model.setParam("NonConvex", 2)
    if args.time_limit:
        opt_model.setParam("TimeLimit", args.time_limit)
    rows, cols = trainX.nonzero()
    # ratings = trainX.data

    # setup decision variables
    u = opt_model.addVars(dataset.n_users,
                          lb=args.lower_bound,
                          ub=args.upper_bound,
                          vtype=grb.GRB.CONTINUOUS,
                          name="users")
    v = opt_model.addVars(dataset.n_movies,
                          lb=args.lower_bound,
                          ub=args.upper_bound,
                          vtype=grb.GRB.CONTINUOUS,
                          name="movies")

    # re-write obj function using constraints, since gurobi does not support mupltication of multiple variables
    objective = 0
    for i, j in zip(rows, cols):
        z = opt_model.addVar(name=f'z_{i}_{j}')
        opt_model.addConstr(u[i] * v[j] == z)
        objective += (z - trainX[i, j]) * (z - trainX[i, j])

    start = time.time()
    # OPTIMIZE
    opt_model.setObjective(objective, grb.GRB.MINIMIZE)
    opt_model.optimize()
    elapsed = time.time() - start
    print(f'Optimization took {elapsed}s')
    # get u and v vectors
    u_arr = np.array([u[i].X for i in range(dataset.n_users)])
    v_arr = np.array([v[i].X for i in range(dataset.n_movies)])
    print('shapes', u_arr.shape, v_arr.shape)
    # keep evaluation schema fixed wrt other experiments
    train_mse = evaluate(u_arr, v_arr, trainX, 'sparse')
    mse = evaluate(u_arr, v_arr, testX, 'sparse')

    als = ALSSparse(u=u_arr, v=v_arr, dataset=trainX)
    fun_eval = als.function_eval()
    print("Final function evaluation", fun_eval)
