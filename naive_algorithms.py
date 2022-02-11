import argparse

from experiment_helper import start_hyper, start_multiple_hyper, start_multi_dataset
from utilities.consts import SINGLE_SEED
from utilities.enums import RecAlgorithmsEnum, RecDatasetsEnum

parser = argparse.ArgumentParser(description='Start an experiment with a naive algorithm')

parser.add_argument('--algorithm', '-a', type=str, help='Recommender System Naive Algorithm',
                    choices=['pop', 'rand'])

parser.add_argument('--dataset', '-d', type=str, help='Recommender System Dataset',
                    choices=[dataset.name for dataset in RecDatasetsEnum])

parser.add_argument('--seed', '-s', type=int, default=SINGLE_SEED, required=False,
                    help='Seed for initialization and sampling')
parser.add_argument('--multiple', '-mp', action='store_true', default=False, required=False,
                    help='Whether to run all the experiments over a single dataset over the specified seeds (see consts.py). Overrides seeds.')
parser.add_argument('--every', '-e', action='store_true', default=False, required=False,
                    help='Whether to run all the experiments over all datasets and all seeds. Overrides multiple.')

args = parser.parse_args()

alg = RecAlgorithmsEnum[args.algorithm] #todo:to change
dataset = RecDatasetsEnum[args.dataset]
seed = args.seed
multiple = args.multiple
every = args.every

if every:
    start_multi_dataset(alg, n_gpus=n_gpus, n_concurrent=n_concurrent, n_samples=n_samples)
elif multiple:
    start_multiple_hyper(alg, dataset, n_gpus=n_gpus, n_concurrent=n_concurrent, n_samples=n_samples)
else:
    start_hyper(alg, dataset, seed, n_gpus=n_gpus, n_concurrent=n_concurrent, n_samples=n_samples)

conf = {
    "data_path": './data/lfm2b-1m',
    'eval_neg_strategy': 'uniform',
    'val_batch_size': 256
}
test_loader = load_data(conf, 'test')

pop_distribution = test_loader.dataset.pop_distribution
top_100 = np.argsort(-pop_distribution)[:100]

hit_ratio_10 = 0
ndcg_10 = 0
for u_idxs, i_idxs, labels in test_loader:
    trues = i_idxs[:, 0]
    for t in trues:
        t = t.item()
        if t in top_100[:10]:
            hit_ratio_10 += 1
            ndcg_10 += 1 / (1 + np.log2(np.where(top_100[:10] == t)[0][0] + 1))

hit_ratio_10 /= test_loader.dataset.n_users
ndcg_10 /= test_loader.dataset.n_users
