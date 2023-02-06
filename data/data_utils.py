import enum

from torch.utils.data import DataLoader

from data.dataloader import TrainDataLoader, NegativeSampler
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
        train_dataset = TrainRecDataset(data_path=conf['dataset_path'])
        sampler = NegativeSampler(
            train_dataset=train_dataset,
            n_neg=conf['neg_train'],
            neg_sampling_strategy=conf['train_neg_strategy']
        )
        dataloader = TrainDataLoader(
            sampler,
            train_dataset,
            batch_size=conf['train_batch_size'],
            shuffle=True,
            num_workers=conf['running_settings']['n_workers'],
            prefetch_factor=conf['running_settings']['prefetch_factor'] if 'prefetch_factor' in conf[
                'running_settings'] else 2
        )
        print(f"Built Train DataLoader module \n"
              f"- batch_size: {conf['train_batch_size']} \n"
              f"- n_workers: {conf['running_settings']['n_workers']} \n")

    elif split_set == 'val':
        dataloader = DataLoader(
            FullEvalDataset(
                data_path=conf['dataset_path'],
                split_set='val',
            ),
            batch_size=conf['eval_batch_size']
        )
        print(f"Built Val DataLoader module \n"
              f"- batch_size: {conf['eval_batch_size']} \n")

    elif split_set == 'test':
        dataloader = DataLoader(
            FullEvalDataset(
                data_path=conf['dataset_path'],
                split_set='test',
            ),
            batch_size=conf['eval_batch_size']
        )
        print(f"Built Test DataLoader module \n"
              f"- batch_size: {conf['eval_batch_size']} \n")
    else:
        raise ValueError(f"split_set value '{split_set}' is invalid! Please choose from [train, val, test]")
    return dataloader
