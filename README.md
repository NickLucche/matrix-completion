# Rank1 ALS Matrix Completion
---
This repo contains the implementation of the Alternating Least Squares algorithm specifically optimized for rank-1 approximation for both sparse and dense scenarios.
Context of the work is that of Collaborative Filtering, so we're looking to "fill gaps in data" by leveraging latent information obtainable through low-rank factorization.
Although in practical situations you're more likely to be in need of a k-rank approximation, the work here has a mainly educational purpose.


# Usage
---
Experiments can be run by executing the `main.py` as in `python main.py -d data/ -s /tmp -u 610 -m 9742`; convergence should be reached in 40+ iterations on the provided sample dataset with default parameters (grad sensibility of `1e-8`).
The script will first execute the sparse version of the algorithm, optionally print some recommendation for a random user (arg `-v`, useful to show inference), followed by the dense version execution.  
Results will be saved as a json file to the directory specified by the save argument `-s`.


# Structure
---
`./data` folder contains the small 100k ratings dataset from [GroupLens](https://grouplens.org/datasets/movielens/), useful for testing and development.
`main.py` contains the code for running and evaluating experiments.
`als.py` contains the actual implementation of the ALS algorithm for both sparse and dense, which are contained in 2 different classes both exposing the same exact interface (`.fit()` for running algorithm plus a few other methods).
`dataset.py` contains a set of utils for loading and handling of a dataset provided by movielens for both sparse and dense/full mode. It also contains code for parallel execution of a test-train split (which can be demanding given a few conditions must be tested before entry extraction) which depends on library `ray` (optional).
`parallel_test_train_split.py` contains code executed by such library.


