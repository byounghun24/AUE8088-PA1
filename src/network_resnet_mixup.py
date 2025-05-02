# Python packages
from termcolor import colored
from typing import Dict
import copy

# PyTorch & Pytorch Lightning
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from torch import nn
from torchvision import models
from torchvision.models.alexnet import AlexNet
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
import torch
import torch.nn.functional as F

# Custom packages
from src.metric import MyAccuracy, MyF1Score
import src.config as cfg
from src.util import show_setting

class SoftCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_logits, soft_labels):
        soft_labels = soft_labels.clamp(min=1e-6)
        log_probs = F.log_softmax(pred_logits, dim=1)
        return -(soft_labels * log_probs).sum(dim=1).mean()

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

        # downsampling for shortcut if needed
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv(x)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)
    
class MyNetwork(nn.Module):
    def __init__(self, num_classes, dropout):
        super().__init__()

        self.features = nn.Sequential(
            ResidualBlock(3, 64, stride=2),
            ResidualBlock(64, 128, stride=1),
            ResidualBlock(128, 192, stride=2),
            ResidualBlock(192, 256, stride=1),
            ResidualBlock(256, 256, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [TODO: Optional] Modify this as well if you want
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class SimpleClassifier(LightningModule):
    def __init__(self,
                 model_name: str = 'resnet18',
                 num_classes: int = 200,
                 optimizer_params: Dict = dict(),
                 scheduler_params: Dict = dict(),
                 dropout: float = 0.3,
        ):
        super().__init__()

        # Network
        if model_name == 'MyNetwork':
            self.model = MyNetwork(num_classes, dropout)
        else:
            models_list = models.list_models()
            assert model_name in models_list, f'Unknown model name: {model_name}. Choose one from {", ".join(models_list)}'
            self.model = models.get_model(model_name, num_classes=num_classes, dropout=dropout)

        # Loss function
        self.loss_fn = SoftCrossEntropy()

        # Metric
        self.accuracy = MyAccuracy()
        self.f1score = MyF1Score(num_classes=num_classes)

        # Hyperparameters
        self.save_hyperparameters()

    def on_train_start(self):
        show_setting(cfg)

    # def configure_optimizers(self):
    #     optim_params = copy.deepcopy(self.hparams.optimizer_params)
    #     optim_type = optim_params.pop('type')
    #     optimizer = getattr(torch.optim, optim_type)(self.parameters(), **optim_params)

    #     scheduler_params = copy.deepcopy(self.hparams.scheduler_params)
    #     scheduler_type = scheduler_params.pop('type')
    #     scheduler = getattr(torch.optim.lr_scheduler, scheduler_type)(optimizer, **scheduler_params)
    #     return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    
    def configure_optimizers(self):
        optim_params = copy.deepcopy(self.hparams.optimizer_params)
        optim_type = optim_params.pop('type')
        optimizer = getattr(torch.optim, optim_type)(self.parameters(), **optim_params)

        scheduler_params = copy.deepcopy(self.hparams.scheduler_params)
        scheduler_type = scheduler_params.pop('type')

        if scheduler_type == 'OneCycleLR':
            scheduler = {
                'scheduler': torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=scheduler_params.pop('max_lr'),
                    steps_per_epoch=self.trainer.estimated_stepping_batches // self.trainer.max_epochs,
                    **scheduler_params
                ),
                'interval': 'step',
                'frequency': 1
            }
        else:
            scheduler = {
                'scheduler': getattr(torch.optim.lr_scheduler, scheduler_type)(optimizer, **scheduler_params),
                'interval': 'epoch',
                'frequency': 1
            }

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, is_train=True)
        accuracy = self.accuracy(scores, torch.argmax(y, dim=1))  # convert to int
        self.log_dict({'loss/train': loss, 'accuracy/train': accuracy}, ...)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, is_train=False)
        accuracy = self.accuracy(scores, y)  # y는 이미 정수형
        self.f1score.update(scores, y)
        self.log_dict({'loss/val': loss, 'accuracy/val': accuracy}, ...)
        self._wandb_log_image(batch, batch_idx, scores, frequency=cfg.WANDB_IMG_LOG_FREQ)


    def on_validation_epoch_end(self):
        f1_per_class = self.f1score.compute()
        macro_f1 = f1_per_class.mean()
        self.log("f1/val_macro", macro_f1, prog_bar=True, logger=True)

    def _common_step(self, batch, is_train=True):
        x, y = batch
        scores = self.forward(x)

        if is_train:
            loss = self.loss_fn(scores, y)  # soft label
        else:
            loss = F.cross_entropy(scores, y)  # hard label

        return loss, scores, y

    def _wandb_log_image(self, batch, batch_idx, preds, frequency = 100):
        if not isinstance(self.logger, WandbLogger):
            if batch_idx == 0:
                self.print(colored("Please use WandbLogger to log images.", color='blue', attrs=('bold',)))
            return

        if batch_idx % frequency == 0:
            x, y = batch
            preds = torch.argmax(preds, dim=1)
            self.logger.log_image(
                key=f'pred/val/batch{batch_idx:5d}_sample_0',
                images=[x[0].to('cpu')],
                caption=[f'GT: {y[0].item()}, Pred: {preds[0].item()}'])
