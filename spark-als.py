from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from pyspark import SparkContext
import numpy as np
from als import ALSSparse

from dataset import MovieLensDataset
from utils import load_matrix, save_matrix, sparse_matrix_to_csv
from argparse import ArgumentParser
from main import evaluate

# spark = SparkSession.builder.appName("ALS").getOrCreate()
sc = SparkContext(appName="RankOneALS")
# sc = SparkContext.getOrCreate()

MAX_ITER = 100
SEED = 10

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument('-d',
                      '--dataset-path',
                      help='Absolute path of the csv dataset to load',
                      required=False)
    args.add_argument('-u',
                      '--n-users',
                      help='Number of users present in the dataset',
                      type=int,
                      required=False)
    args.add_argument('-m',
                      '--n-movies',
                      help='Number of movies present in the dataset',
                      type=int,
                      required=False)
    args.add_argument('-i',
                      '--n-iterations',
                      help='Number of iterations to run ALS algorithm for',
                      type=int,
                      default=1000)
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

    try:
        print("Loading train and test split from /tmp/..")
        trainX = load_matrix(f'trainX_sparse', True)
        testX = load_matrix(f'testX_sparse', True)
    except:
        print("Loading failed, generating train-test split now..")
        dataset = MovieLensDataset(args.dataset_path, mode='sparse')
        # %5 test size
        test_set_size = dataset.n_ratings // 20
        trainX, testX = dataset.train_test_split_simple(test_set_size)
        print(f"Saving train and test set to /tmp/ first..")
        save_matrix(f'trainX_sparse', trainX)
        save_matrix(f'testX_sparse', testX)

    # train matrix to csv
    print("Storing train matrix to csv..")
    sparse_matrix_to_csv('/tmp/trainX.csv', trainX)
    # testX = testX.toarray()
    data = sc.textFile('/tmp/trainX.csv')
    ratings = data.map(lambda l: l.split(','))\
        .map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))

    # Build the recommendation model using Alternating Least Squares
    rank = 1
    # do this otherwise ALS crashes <.<
    sc.setCheckpointDir('/tmp/')
    print("Running ALS..")
    model = ALS.train(ratings,
                      rank,
                      args.n_iterations,
                      lambda_=0.0,
                      nonnegative=args.non_negative,
                      seed=SEED)

    # get u, v user and items vectors
    pf = model.productFeatures()
    v = np.matrix(np.asarray(pf.values().collect()).astype(np.float64))
    print('V:', v.shape, np.linalg.norm(v), v.max(), v.min())

    pf = model.userFeatures()
    u = np.matrix(np.asarray(pf.values().collect()).astype(np.float64))
    print('U:', u.shape, np.linalg.norm(u), u.max(), u.min())

    # keep evaluation schema fixed wrt other experiments
    mse = evaluate(u, v, testX, 'sparse')

    sparse_matrix_to_csv('/tmp/testX.csv', testX)
    # testX = testX.toarray()
    data = sc.textFile('/tmp/testX.csv')
    ratings = data.map(lambda l: l.split(','))\
        .map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))

    testdata = ratings.map(lambda p: (p[0], p[1]))
    predictions = model.predictAll(testdata).map(lambda r:
                                                 ((r[0], r[1]), r[2]))
    ratesAndPreds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(
        predictions)
    MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
    print("Mean Squared Error = " + str(MSE))

    # compare with what i did so far
    # dataset = MovieLensDataset('/tmp/', n_users=args.n_users, n_movies=args.n_movies, mode='sparse')
    # from main import init_vector
    # u = init_vector(dataset.n_users, normalize=True)
    # v = init_vector(dataset.n_movies, normalize=True)
    # trainX, testX = dataset.train_test_split_simple(1000//20)
    # args = [u, v, trainX]
    # als = ALSSparse(*args)
    # u, v = als.fit()

    # # dataset = MovieLensDataset('/tmp/testX.csv', n_users=args.n_users, n_movies=args.n_movies, mode='sparse')
    # mse = evaluate(u, v, trainX, 'sparse')
