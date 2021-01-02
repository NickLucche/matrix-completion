from pyspark.context import SparkContext
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.sql import SparkSession

from dataset import MovieLensDataset
from utils import load_matrix, save_matrix
from argparse import ArgumentParser

spark = SparkSession.builder.appName("ALS").getOrCreate()
sc = SparkContext.getOrCreate()

MAX_ITER = 100
SEED = 10

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument('-d', '--dataset-path', help='Absolute path of the csv dataset to load', required=True)
    args.add_argument('-s', '--save-path', help='Directory where to save factorization results to', default='./data/')
    args.add_argument('-u', '--n-users', help='Number of users present in the dataset', type=int, required=True)
    args.add_argument('-m', '--n-movies', help='Number of movies present in the dataset', type=int, required=True)
    args.add_argument('-w', '--n-workers', help='Number of workers used to split dataset into test-train', type=int, default=8)
    args = args.parse_args()

    dataset = MovieLensDataset(args.dataset_path, n_users=args.n_users, n_movies=args.n_movies, mode='full')

    # lines = spark.read.text("/home/nick/Downloads/spark/data/mllib/als/sample_movielens_ratings.txt").rdd
    # parts = lines.map(lambda row: row.value.split("::"))
    # ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),
                                        # rating=float(p[2]), timestamp=p[3]))
    # ratings = spark.createDataFrame(ratingsRDD)
    # (training, test) = ratings.randomSplit([0.8, 0.2])
    
    try:
        print("Loading train and test split from /tmp/..")
        trainX = load_matrix(f'trainX_full', False)
        testX = load_matrix(f'testX_full', False)
    except:
        print("Loading failed, generating train-test split now..")
        # %5 test size
        test_set_size = dataset.n_ratings//20
        trainX, testX = dataset.train_test_split(test_set_size, args.n_workers)
        print(f"Saving train and test set to /tmp/ first..")
        save_matrix(f'trainX_full', trainX)
        save_matrix(f'testX_full', testX)
    
    testX = testX.toarray()
    # convert dataset to pyspark dataframe
    rows, cols = trainX.nonzero()
    train_list = [[user, movie, trainX[user, movie]] for user, movie in zip(rows, cols)]
    rdd1 = sc.parallelize(train_list)
    ratingsRDD = rdd1.map( lambda p: Row(userId=int(p[0]), movieId=int(p[1]), rating=float(p[2]) ))
    train_df = spark.createDataFrame(ratingsRDD)
    # train_df = rdd2.toDF(["userId", "movieId", "rating"])
    print(train_df.show())
    # rdd1 = sc.parallelize(testX)
    # rdd2 = rdd1.map(lambda x: [int(i) for i in x])
    # test_df = rdd2.toDF([str(i) for i in range(dataset.n_movies)])

    # TODO: match format to the one you have here /home/nick/Downloads/spark/data/mllib/als/sample_movielens_ratings.txt
    # ratings = spark.read.option("header", "true").csv("ml-20m/ratings.csv")

    # Build the recommendation model using ALS on the training data
    # Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
    als = ALS(maxIter=MAX_ITER, rank=1, regParam=0, userCol="userId", itemCol="movieId", ratingCol="rating", seed=SEED)
    model = als.fit(train_df)

    # Evaluate the model by computing the RMSE on the test data
    # predictions = model.transform(train_df)
    # evaluator = RegressionEvaluator(metricName="mse", labelCol="rating",
    #                                 predictionCol="prediction")
    # rmse = evaluator.evaluate(predictions)
    # print("Mean-square error = " + str(rmse))

    # Generate top 10 movie recommendations for each user
    # userRecs = model.recommendForAllUsers(10)
    # Generate top 10 user recommendations for each movie
    # movieRecs = model.recommendForAllItems(10)