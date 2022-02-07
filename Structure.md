# Hassaku
Folder structure

# algorithms:
Contains the algorithms currently implemented divided by category

# data:
contains the dataset and the dataset.py class to handle datasets

# utilities:
- consts.py: contains constants such as max number of epochs, data path etc.
- enums.py: contains enumerators for both algorithms and datasets
- eval.py: contains metrics, evaluator, and a general evaluation of recommender system algorithms
- rec_losses.py: contains recommender system losses used by sgd-based rec algs
- similarities.py: contains similarity functions used by knn
- trainer.py: contains the class used to train sgd-based rec algs
- utils.py: miscellaneous

# experiment_helper.py:
main functionalities of the framework regarding the hyperparameter optimiation

# hyper_params.py:
hyperparameters used by the algorithms

# start.py:
starting point of the framework