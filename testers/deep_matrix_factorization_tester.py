import sys

sys.path.append('../')
sys.path.append('../algorithms')
sys.path.append('../utilities')
from data.dataset import get_recdataset_dataloader
from utilities.eval import evaluate_recommender_algorithm

SEED = 1391075

train_loader = get_recdataset_dataloader(
    'inter',
    data_path='../data/lfm2b-1m',
    split_set='train',
    n_neg=10,
    neg_strategy='uniform',
    batch_size=64,
    shuffle=True,
    num_workers=8,
)
val_loader = get_recdataset_dataloader(
    'inter',
    data_path='../data/lfm2b-1m',
    split_set='val',
    n_neg=99,
    neg_strategy='uniform',
    batch_size=64,
    num_workers=8,
)

test_loader = get_recdataset_dataloader(
    'inter',
    data_path='../data/lfm2b-1m',
    split_set='test',
    n_neg=99,
    neg_strategy='uniform',
    batch_size=64,
    num_workers=8,
)

from algorithms.neural_alg import DeepMatrixFactorization
from utilities.trainer import ExperimentConfig, Trainer

dmf_alg = DeepMatrixFactorization(train_loader.dataset.iteration_matrix, u_mid_layers=[100], i_mid_layers=[100],
                                  final_dimension=64)
conf = ExperimentConfig(lr=1e-4)
trainer = Trainer(dmf_alg, train_loader, val_loader, conf)
dmf_alg = trainer.fit()

evaluate_recommender_algorithm(dmf_alg, test_loader, SEED)
