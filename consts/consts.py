# --- Data Processing Constants --- #
INF_STR = "{:10d} entries {:7d} users {:7d} items for {}"
LOG_FILT_DATA_PATH = "log_filtering_data.txt"

# --- Experiment Constants --- #
SINGLE_SEED = 38210573
SEED_LIST = [SINGLE_SEED, 9491758, 2931009]
NUM_SAMPLES = 50
NUM_EPOCHS = 50

# --- Evaluation Constants --- #
K_VALUES = [1, 3, 5, 10, 20, 50, 100]  # K value for the evaluation metrics
EVAL_BATCH_SIZE = 32
OPTIMIZING_METRIC = 'ndcg@10'  # Which metric will be used to assess during validation.
# --- Logger Constants --- #
ENTITY_NAME = 'jku-mms'
PROJECT_NAME = 'advprotomf'

# --- Path Constants --- #
DATA_PATH = 'data'
WANDB_API_KEY_PATH = './other/wandb_api_key'
