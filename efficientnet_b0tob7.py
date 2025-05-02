import argparse
import os
import torch
from lightning import Trainer
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

# Custom packages
from src.dataset import TinyImageNetDatasetModule
from src.network import SimpleClassifier
import src.config as cfg

torch.set_float32_matmul_precision('medium')

MODEL_SETTINGS = {
    'efficientnet_b0': {'dropout': 0.2,  'max_lr': 0.05,  'epochs': 40,  'batch_size': 512},
    'efficientnet_b1': {'dropout': 0.25, 'max_lr': 0.05,  'epochs': 50,  'batch_size': 512},
    'efficientnet_b2': {'dropout': 0.3,  'max_lr': 0.05,  'epochs': 60,  'batch_size': 512},
    'efficientnet_b3': {'dropout': 0.35, 'max_lr': 0.04,  'epochs': 80,  'batch_size': 512},
    'efficientnet_b4': {'dropout': 0.4,  'max_lr': 0.035, 'epochs': 100, 'batch_size': 512},
    'efficientnet_b5': {'dropout': 0.45, 'max_lr': 0.03,  'epochs': 120, 'batch_size': 512},
    'efficientnet_b6': {'dropout': 0.5,  'max_lr': 0.025, 'epochs': 150, 'batch_size': 512},
    'efficientnet_b7': {'dropout': 0.5,  'max_lr': 0.02,  'epochs': 200, 'batch_size': 512},
}


def get_augmentation_level(model_name: str):
    if 'b0' in model_name or 'b1' in model_name or 'b2' in model_name:
        return 1
    elif 'b3' in model_name or 'b4' in model_name:
        return 2
    else:
        return 3  # b5 ~ b7

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default=cfg.MODEL_NAME, help="Model name to use")
    args = parser.parse_args()

    cfg.MODEL_NAME = args.model_name.lower()

    if cfg.MODEL_NAME in MODEL_SETTINGS:
        s = MODEL_SETTINGS[cfg.MODEL_NAME]
        cfg.NUM_EPOCHS = s['epochs']
        cfg.BATCH_SIZE = s['batch_size']
        cfg.OPTIMIZER_PARAMS['lr'] = s['max_lr']
        cfg.SCHEDULER_PARAMS.update({
            'max_lr': s['max_lr'],
            'epochs': s['epochs'],
        })
        cfg.DROPOUT = s['dropout']
        cfg.AUG_LEVEL = get_augmentation_level(cfg.MODEL_NAME)
    else:
        raise ValueError(f"Unsupported model: {cfg.MODEL_NAME}")

    cfg.WANDB_NAME = f'{cfg.MODEL_NAME}-B{cfg.BATCH_SIZE}-{cfg.OPTIMIZER_PARAMS["type"]}'
    cfg.WANDB_NAME += f'-{cfg.SCHEDULER_PARAMS["type"]}{cfg.OPTIMIZER_PARAMS["lr"]:.1E}'

    model = SimpleClassifier(
        model_name = cfg.MODEL_NAME,
        num_classes = cfg.NUM_CLASSES,
        optimizer_params = cfg.OPTIMIZER_PARAMS,
        scheduler_params = cfg.SCHEDULER_PARAMS,
        dropout = cfg.DROPOUT
    )

    datamodule = TinyImageNetDatasetModule(
        batch_size = cfg.BATCH_SIZE,
    )

    wandb_logger = WandbLogger(
        project = cfg.WANDB_PROJECT,
        save_dir = cfg.WANDB_SAVE_DIR,
        entity = cfg.WANDB_ENTITY,
        name = cfg.WANDB_NAME,
    )

    trainer = Trainer(
        accelerator = cfg.ACCELERATOR,
        devices = cfg.DEVICES,
        precision = cfg.PRECISION_STR,
        max_epochs = cfg.NUM_EPOCHS,
        check_val_every_n_epoch = cfg.VAL_EVERY_N_EPOCH,
        logger = wandb_logger,
        callbacks = [
            LearningRateMonitor(logging_interval='epoch'),
            ModelCheckpoint(save_top_k=1, monitor='accuracy/val', mode='max'),
        ],
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.validate(ckpt_path='best', datamodule=datamodule)
