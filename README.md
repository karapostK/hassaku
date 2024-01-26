# :tangerine: Hassaku - A Recommender Systems Framework for Research

<div align="center">
    <img src="./assets/hassaku.png" style="width: 320px" />
</div>

## What is Hassaku?

Beyond being a [Japanese citrus hybrid](https://en.wikipedia.org/wiki/Hassaku_orange), **Hassaku** is a(nother)
research-oriented Recommender System Framework written mostly in Python and leveraging PyTorch.
**Hassaku** hosts the main procedure for data processing, training, testing, hyperparameter optimization, metrics
computation, and logging for several Collaborative Filtering-based Recommender Systems.
The project mostly started as a way to support my own research, however, it is open to everyone who wants to experiment
with Recommender Systems algorithms and run exciting experiments.

### What does Hassaku have?

- Basic Train/Val/Test strategies for a Recommender Systems based on implicit feedback
- Negative Sampling strategies
- Several Collaborative Filtering algorithms
- Full evaluation on all items during validation and test
- Automatic sync to [Wandb](https://wandb.ai/site)
- Hyperparameter Optimization offered by [Ray Tune](https://www.ray.io/ray-tune)

### What doesn't Hassaku have?

- A wide selection of algorithms and options compared to [RecBole](https://www.recbole.io/)
  and [Elliot](https://elliot.readthedocs.io/en/latest/) (I am one person after all :adult:)

## Installation and Configuration

### Environment

- Install the environment with
  `conda env create -f hassaku.yml`
- Activate the environment with `conda activate hassaku`

### Data

Generally, **Hassaku** works with user-based ratio-based train/val/test splits of users'
interactions ([this paper offers nice brief description about it](https://dl.acm.org/doi/pdf/10.1145/3340531.3412095)),
sometimes named _weak generalization_.
The folder `./data/` hosts the code for downloading, filtering, and generally processing some RecSys datasets. There are
several default `*_processor.py` in each dataset folder that perform temporal-based splits of the users-items
interactions.

E.g. to download and process the [MovieLens 1M Dataset](https://grouplens.org/datasets/movielens/1m/):

- move into the folder with `cd data/ml1m`
- run `python movielens1m_processor.py` (if the dataset is not detected, it will be automatically downloaded)

After the script ends you will have 5 files in the newly created `processed_dataset` folder:

- 3 files with the listening_history of the users for train, val, and test
- 2 files containing the user and item ids (as index in the rating matrix)

### Wandb configuration

Place your [wandb api key](https://wandb.ai/authorize) in a file, e.g., `./wandb_api_key`.

In `wandb_conf.py` set:

- `PROJECT_NAME` = `hassaku` (or something else if you want)
- `ENTITY_NAME` = `<your_name_on_wandb>`
- `WANDB_API_KEY_PATH` = `./wandb_api_key`

## Quick Start

Showcasing how to run one or multiple experiments with **Hassaku**,
considering [BPRMF](https://arxiv.org/pdf/1205.2618.pdf) applied on MovieLens 1M

### One experiment

1) Run `movielens1m_processor.py` in `./data/ml1m/` (as in "Installation and Configuration/Data")
2) Set all your experiment configurations in a .yml file in the e.g., `./conf` folder. For example, `bprmf_conf.yml`:

```yaml
data_path: ./data # or absolute path

# Model Parameters
embedding_dim: 402
lr: 0.0003
wd: 0.00004
use_user_bias: False
use_item_bias: True
use_global_bias: False

# Training Parameters
optimizer: adamw
n_epochs: 50
max_patience: 5
train_batch_size: 128
neg_train: 50
rec_loss: bpr

# Experiment Parameters
eval_batch_size: 256
device: cuda
running_settings:
  train_n_workers: 4
  batch_verbose: True
```

3. Run `python run_experiment -a mf -d ml1m -c ./conf/bprmf_conf.yml` (This will perform both train/val and test)
4. Track your run on wandb.

### Hyperparameter Optimization

1) Run `movielens1m_processor.py` in `./data/ml1m/` (as in "Installation and Configuration/Data")
2) Write your hyperparameter ranges and set parameters in `./hyper_search/hyper_params.py` (see [Ray Tune](https://docs.ray.io/en/latest/tune/api/search_space.html#tune-search-space) docs about this)

```python
bprmf_ml1m_param = {
    # Hyper Parameters
    'lr': tune.loguniform(5e-5, 1e-3),
    'wd': tune.loguniform(1e-6, 1e-2),
    'embedding_dim': tune.lograndint(126, 512, base=2),
    'train_batch_size': tune.lograndint(32, 256, base=2),
    'neg_train': tune.randint(1, 100),
    # Set Parameters
    'n_epochs': 50,
    'max_patience': 5,
    'running_settings':
        {
            'train_n_workers': 4
        },
    'train_neg_strategy': 'uniform',
    'rec_loss': 'bpr',
    'use_user_bias': False,
    'use_item_bias': True,
    'use_global_bias': False,
    'optimizer': 'adamw'
}
```

and make sure that it will be selected (bottom of `./hyper_search/hyper_params.py`)

```python
alg_data_param = {
...
(AlgorithmsEnum.mf, DatasetsEnum.ml1m): bprmf_ml1m_param,
...
}
```

3) Run `python run_hyper_experiment -a mf -d ml1m -dp <path_to_the_./data_directory` \\

Optionally, you can set the number of sampled configurations by `-ns` (e.g., `-ns 50`), cpus and gpus for each trial, <=1 values are allowed, using `-ncpu -ngpu` (e.g., `-ncpu 1 -ngpu 0.2`), and the number of concurrent trial with `-nc` (e.g., `-nc 3`).

4) Track all of your trials on wandb
