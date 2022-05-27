import enum

from torch.utils.data import DataLoader

from data.dataset import TrainRecDataset, FullEvalDataset

INF_STR = "{:10d} entries {:7d} users {:7d} items for {}"
LOG_FILT_DATA_PATH = "log_filtering_data.txt"


# Note that the name of the dataset should correspond to a folder in data. e.g. ml1m has a corresponding folder in /data
class DatasetsEnum(enum.Enum):
    ml1m = enum.auto()
    ml10m = enum.auto()
    amazonvid = enum.auto()
    lfm2b2020 = enum.auto()


def print_and_log(log_file, n_lhs, n_users, n_items, text):
    info_string = INF_STR.format(n_lhs, n_users, n_items, text)
    log_file.write(info_string + '\n')
    print(info_string)


def get_dataloader(conf: dict, split_set: str) -> DataLoader:
    """
        Returns the dataloader associated to the configuration in conf
    """

    if split_set == 'train':
        return DataLoader(
            TrainRecDataset(
                data_path=conf['dataset_path'],
                n_neg=conf['neg_train'],
                neg_sampling_strategy=conf['train_neg_strategy'],
            ),
            batch_size=conf['train_batch_size'],
            shuffle=True,
            num_workers=conf['running_settings']['n_workers']
        )
    elif split_set == 'val':
        return DataLoader(
            FullEvalDataset(
                data_path=conf['dataset_path'],
                split_set='val',
            ),
            batch_size=conf['eval_batch_size']
        )

    elif split_set == 'test':
        return DataLoader(
            FullEvalDataset(
                data_path=conf['dataset_path'],
                split_set='test',
            ),
            batch_size=conf['eval_batch_size']
        )
