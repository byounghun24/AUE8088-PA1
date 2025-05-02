import os

# Training Hyperparameters
NUM_CLASSES         = 200
BATCH_SIZE          = 512
VAL_EVERY_N_EPOCH   = 1

NUM_EPOCHS          = 70

OPTIMIZER_PARAMS = {
    'type': 'SGD',
    'lr': 0.03,             # ← 증가
    'momentum': 0.9,
    'weight_decay': 0.0005
}

SCHEDULER_PARAMS    = {
    'type': 'OneCycleLR',
    'max_lr': 0.03,
    'epochs': NUM_EPOCHS,
    'pct_start': 0.15,
    'anneal_strategy': 'cos',
    'div_factor': 25.0,       # initial_lr = max_lr / div_factor
    'final_div_factor': 1e4   # eta_min = max_lr / final_div_factor
}

DROPOUT = 0.1

# Dataaset
DATASET_ROOT_PATH   = 'datasets/'
NUM_WORKERS         = 8

# Augmentation
IMAGE_ROTATION      = 0
IMAGE_FLIP_PROB     = 0.5
IMAGE_NUM_CROPS     = 64
IMAGE_PAD_CROPS     = 4
IMAGE_MEAN          = [0.4802, 0.4481, 0.3975]
IMAGE_STD           = [0.2302, 0.2265, 0.2262]

# Network
MODEL_NAME          = 'MyNetwork'

# Compute related
ACCELERATOR         = 'gpu'
DEVICES             = [0]
PRECISION_STR       = '32-true'

# Logging
WANDB_PROJECT       = 'aue8088-pa1'
WANDB_ENTITY        = os.environ.get('WANDB_ENTITY')
WANDB_SAVE_DIR      = 'wandb/'
WANDB_IMG_LOG_FREQ  = 50
WANDB_NAME          = f'{MODEL_NAME}-B{BATCH_SIZE}-{OPTIMIZER_PARAMS["type"]}'
WANDB_NAME         += f'-{SCHEDULER_PARAMS["type"]}{OPTIMIZER_PARAMS["lr"]:.1E}'
