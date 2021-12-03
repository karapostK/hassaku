import sys

sys.path.append('../')
sys.path.append('../algorithms')
from data.dataset import get_recdataset_dataloader
from utilities.eval import Evaluator
from utilities.utils import reproducible, print_results

SEED = 1391075

train_loader = get_recdataset_dataloader(
    data_path='../data/lfm2b-1m',
    split_set='train',
    n_neg=10,
    neg_strategy='uniform',
    batch_size=64,
    shuffle=True,
    num_workers=8,
)
val_loader = get_recdataset_dataloader(
    data_path='../data/lfm2b-1m',
    split_set='val',
    n_neg=99,
    neg_strategy='uniform',
    batch_size=64,
    num_workers=8,
)

test_loader = get_recdataset_dataloader(
    data_path='../data/lfm2b-1m',
    split_set='test',
    n_neg=99,
    neg_strategy='uniform',
    batch_size=64,
    num_workers=8,
)


def evaluate_rec_alg(rec_alg, test_loader):
    reproducible(SEED)

    evaluator = Evaluator(test_loader.dataset.n_users)

    for u_idxs, i_idxs, labels in test_loader:
        out = rec_alg.predict(u_idxs, i_idxs)

        evaluator.eval_batch(out)

    metrics_values = evaluator.get_results()
    print_results(metrics_values)


from algorithms.neural_alg import DeepMatrixFactorization
from utilities.trainer import ExperimentConfig, Trainer

dmf_alg = DeepMatrixFactorization(train_loader.dataset.csr_matrix, u_mid_layers=[100], i_mid_layers=[100],
                                  final_dimension=32)
conf = ExperimentConfig(lr=1e-4)
trainer = Trainer(dmf_alg, train_loader, val_loader, conf)
dmf_alg = trainer.fit()

evaluate_rec_alg(dmf_alg, test_loader)
