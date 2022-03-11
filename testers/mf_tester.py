import sys

from torch.utils.data import DataLoader

sys.path.append('../')
sys.path.append('../algorithms')
sys.path.append('../utilities')
from data.dataset import get_recdataset_dataloader
from algorithms.base_classes import RecommenderAlgorithm
from utilities.consts import SINGLE_SEED
from utilities.utils import reproducible, print_results
from utilities.eval import Evaluator

train_loader = get_recdataset_dataloader(
    'inter',
    data_path='../data/lfm2b1m',
    split_set='train',
    n_neg=10,
    neg_strategy='uniform',
    batch_size=64,
    shuffle=True,
    num_workers=8,
)

val_loader = get_recdataset_dataloader(
    'inter',
    data_path='../data/lfm2b1m',
    split_set='val',
    n_neg=99,
    neg_strategy='uniform',
    batch_size=64,
    num_workers=8,
)

test_loader = get_recdataset_dataloader(
    'inter',
    data_path='../data/lfm2b1m',
    split_set='test',
    n_neg=99,
    neg_strategy='uniform',
    batch_size=64,
    num_workers=8,
)


def evaluate_recommender_algorithm(alg: RecommenderAlgorithm, eval_loader: DataLoader, seed: int = SINGLE_SEED):
    reproducible(seed)

    evaluator = Evaluator(eval_loader.dataset.n_users)

    for u_idxs, i_idxs, labels in eval_loader:
        out = alg.predict(u_idxs, i_idxs)

        evaluator.eval_batch(out)

    metrics_values = evaluator.get_results()
    print_results(metrics_values)


from algorithms.neural_alg import SGDMatrixFactorization
from utilities.trainer import ExperimentConfig, Trainer

mf_alg = SGDMatrixFactorization(train_loader.dataset.n_users, train_loader.dataset.n_items, 64)
conf = ExperimentConfig(lr=1e-3, device='cpu')
trainer = Trainer(mf_alg, train_loader, val_loader, conf)
mf_alg = trainer.fit()

evaluate_recommender_algorithm(mf_alg, test_loader)
