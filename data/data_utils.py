import enum

INF_STR = "{:10d} entries {:7d} users {:7d} items for {}"
LOG_FILT_DATA_PATH = "log_filtering_data.txt"


# Note that the name of the dataset should correspond to a folder in data. e.g. ml1m has a corresponding folder in /data
class RecDatasetsEnum(enum.Enum):
    ml1m = enum.auto()
    ml10m = enum.auto()
    amazonvid = enum.auto()
    lfm2b2020 = enum.auto()


def print_and_log(log_file, n_lhs, n_users, n_items, text):
    info_string = INF_STR.format(n_lhs, n_users, n_items, text)
    log_file.write(info_string + '\n')
    print(info_string)
