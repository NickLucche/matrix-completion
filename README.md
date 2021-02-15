# Rank1 ALS Matrix Completion
---
This repo contains the implementation of the Alternating Least Squares algorithm specifically optimized for rank-1 approximation for both sparse and dense scenarios.
Context of the work is that of Collaborative Filtering, so we're looking to "fill gaps in data" by leveraging latent information obtainable through low-rank factorization.
Although in practical situations you're more likely to be in need of a k-rank approximation, the work here has a mainly educational purpose.

Experiments validated superior performances wrt other frameworks (BFGS from `scipy.optimize` and the ALS implementation from `Apache Spark`) in optimizing the squared error function for rank-1 factorization of a given sparse data matrix X. Optimization loop has a cost linearly dependant on number of non-zero entries in X (ratings) while the number of iterations performed does not depend on size of data (like GD), so one can expect a consistent behavior in solving problems with 100k or 20M ratings (fixing gradient sensibility and distance from optimum/~"initialization scheme").

![Data growth scaling](/assets/datagrows.png)
![SparseVDense](/assets/sparsevdense_bars.png)



# Usage
---
Experiments can be run by executing the `main.py` as in `python main.py -d data/ -s /tmp`; convergence should be reached in 40+ iterations on the provided 100k ratings dataset with default parameters (grad sensibility of `1e-8`). More info on allowed arguments with `python main.py --help`.

The script will first execute the sparse version of the algorithm, optionally print some recommendation for a random user (arg `-v`, useful to show inference), followed by the dense version execution (only if `--dense` option is passed).  
Resulting factorization vectors `u, v` will be stored in the directory specified by the save argument `-s`.

Mind intermediate results such as train-test splits are saved to `/tmp` for saving up time between multiple trials; make sure to remove those `*.npy/npz` files when testing out execution on a different dataset. 


# Structure
---
`./data` folder contains the 100k ratings dataset from [GroupLens](https://grouplens.org/datasets/movielens/), useful for testing and development.
`main.py` contains the code for running and evaluating ALS algorithm experiments.
`als.py` contains the actual implementation of the ALS algorithm for both sparse and dense, which are contained in 2 different classes both exposing the same exact interface (`.fit()` for running algorithm plus a few extra methods). Major "components" wrapped in the class algorithm are the computation of the optimizer, of the partial derivatives and of the function. 
`dataset.py` contains a class wrapping a set of utils for loading and handling of a dataset provided by movielens for both sparse and dense/full mode. It also contains code for parallel execution of a test-train split (which can be demanding given a few conditions must be tested before entry extraction) which depends on library `ray` (optional).
`parallel_test_train_split.py` contains code executed by such library.
`spark-als.py` runs the Apache Spark general version of the ALS algorithm. By default algorithm is executed by ignoring the regularization term (`lambda_`) and with `rank=1` for optimally comparing with our implementation.
`gurobi-experiments.py` is a script for solving the same problem using the global quadratic solver featured in Gurobi for comparison. Mind you'll probably need a license to run it. Execution may very well take up hours (also it uses increasingly more RAM).   


