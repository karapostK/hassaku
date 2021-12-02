import sys

sys.path.append('../')
sys.path.append('../algorithms')
from algorithms.neural_alg import DeepMatrixFactorization
from data.dataset import get_protorecdataset_dataloader
from utilities.eval import Evaluator
from utilities.utils import reproducible, print_results

SEED = 1391075

train_loader = get_protorecdataset_dataloader(
    data_path='../data/lfm2b-1m',
    split_set='train',
    n_neg=10,
    neg_strategy='uniform',
    batch_size=64,
    shuffle=True,
    num_workers=8,
)
val_loader = get_protorecdataset_dataloader(
    data_path='../data/lfm2b-1m',
    split_set='val',
    n_neg=99,
    neg_strategy='uniform',
    batch_size=64,
    num_workers=8,
)

test_loader = get_protorecdataset_dataloader(
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


dmf_alg = DeepMatrixFactorization(train_loader.dataset.csr_matrix, u_mid_layers=[1000], i_mid_layers=[1000],
                                  final_dimension=32)
dmf_alg.fit(train_loader, n_epochs=200, lr=1e-4)
