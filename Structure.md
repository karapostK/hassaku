# Hassaku
Folder structure
```
.
├── algorithms
├── conf
├── data
├── eval
├── explanations
├── hyper_saved_models
├── hyper_search
├── saved_models
├── framework_tests
├── train
├── utilities
├── README.md
├── Structure.md
├── experiment_helper.py
├── run_baselines.py
├── run_experiment.py
├── run_hyper_baselines.py
├── run_hyper_experiment.py
├── hassaku.yml
├── wandb_api_key
└── wandb_conf.py
```

### ```algorithms```
Hosts the code of the implemented algorithms. Roughly divided into classes.

```
.
├── algorithms_utils.py
├── base_classes.py
├── graph_algs.py
├── knn_algs.py
├── linear_algs.py
├── mf_algs.py
├── naive_algs.py
└── sgd_alg.py
```

### ```conf```
Directory to host the configuration for the experiments. 
```
.
├── <here you can place your .yml files>
└── conf_parser.py
```

### ```data```
Directory to host raw dataset, processed dataset, dataloaders, dataset classes, and dataset processing code. 
```
.
├── amazonvid2018
│   ├── processed_dataset
│   ├── raw_dataset
│   └── amazonvid2018_processor.py
├── lfm2b2020
│   ├── processed_dataset
│   ├── raw_dataset
│   └── lfm2b2020_processor.py
├── ml10m
│   ├── processed_dataset
│   ├── raw_dataset
│   └── movielens10m_processor.py
├── ml1m
│   ├── processed_dataset
│   ├── raw_dataset
│   └── movielens1m_processor.py
├── dataloader.py
├── dataset.py
└── data_utils.py

```

### ```eval```
Directory to host the evaluation metrics and evaluation procedure 
```
.
├── eval.py
├── eval_utils.py
└── metrics.py

```

### ```explanations```
Directory to host code for generating explanations 
```
.
└── utils.py
```


### ```hyper_search```
Directory to host the main code for hyperparameter search, and hyperparameters.
```
.
├── experiment_helper.py
├── hyper_params.py
└── utils.py
```

### ```hyper_saved_model```
Directory to host the results of the hyperparameter search. The folder in this directory are automatically created by ray tune. 
Example below:
```
.
└── uknn-ml1m                                                   # Algortihm + Dataset
    └── 2023-3-10_16-9-57.854261                                # Timestamp of the hyperparameter search
        ├── 2023-3-10_16-10-0.156586_32e75bb5                   # Timestamp of the trail + id 
        │   ├── events.out.tfevents.1678461000.passionpit.cp.jku.at
        │   ├── params.json
        │   ├── params.pkl
        │   ├── progress.csv
        │   ├── result.json 
        │   ├── model.npz                                 # Model 
        │   └── wandb
        ...
```

### ```saved_models```
Directory to host the results of the experiments (not hyperparameters). The folder in this directory are automatically created by my code. 
Example below:
```
.
├── pop-ml1m                                                    #  Algortihm + Dataset
│   ├── 2023-3-8_14-51-27.802868
│   │   └── conf.yml                                # Conf of the experiment 
                                                                # The model is also saved here when needed
│   └── 2023-3-8_16-50-48.826040
│       └── conf.yml
...
```

### ```framework_tests```
Directory to host the code to assess the functionalities of the framework (e.g. the metrics)
```
.
├── data
└── eval
    └── test_metrics.py
```
### ```train```
Directory to host the code to train a SGD-based recommendere system
```
.
├── rec_losses.py
├── trainer.py
└── utils.py
```

### ```utilities```
Directory to host the miscellaneous code.
```
.
├── similarities.py
└── utils.py
```