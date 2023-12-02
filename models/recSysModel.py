"""
This script contains the definition of the classifier based on top of the autoencoder 
"""

import os
import sys
import pytorch_lightning as L
import torch
import torch.nn.functional as F
import torchvision.transforms as tr
import random
import shutil
import wandb
import pandas as pd

random.seed(69)
torch.manual_seed(69)

from torch import optim
from pathlib import Path
from typing import Any, Union, Sequence, Tuple, Dict
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import nn

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything
from torchvision import transforms as trn
from pytorch_lightning.callbacks import ModelCheckpoint

seed_everything(69, workers=True)

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
current = SCRIPT_DIR

while 'models' not in os.listdir(current):
    current = Path(current).parent

PARENT_DIR = str(current)
sys.path.append(str(current))
DATA_FOLDER = os.path.join(current, 'data')
print(DATA_FOLDER)


WANDB_PROJECT_NAME = 'recommendatation_system'

import utilities.directories_and_files as dirf
from models.classification_head import ExponentialClassifier
from models.building_blocks import LinearBlock
from models.recSysDataset import RecSysDataset, USER_DATA_COLS, ITEM_DATA_COLS

class RecSys(L.LightningModule):
    def __init__(self, 
                 num_users: int,
                 num_items: int,
                 context_length: int,
                 emb_dim: int,
                 num_context_blocks: int,
                 num_features_blocks: int,
                 coeff: float = 0.5,
                 learning_rate: float = 10 ** -4,  
                 gamma: float = 0.99,
                 dropout: Union[float, Sequence[float]] = None,
                 *args, 
                 **kwargs
                 ):
        
        self.n_users = num_users
        self.n_items = num_items
        self.emb_dim = emb_dim
        self.coeff = coeff

        # the first step is to create 2 mebedding layers
        # one for the users and one of the moveis
        self.user_emb_layer = nn.Embedding(num_embeddings=num_users, 
                                           embedding_dim=emb_dim)

        self.item_emb_layer = nn.Embedding(num_embeddings=num_items, 
                                           embedding_dim=emb_dim)

        # create the fully connected block that will map the context features into a 2 * n 
        self.context_block = ExponentialClassifier(num_classes=2 * self.emb_dim,
                                                   in_features=context_length, 
                                                   num_layers=num_context_blocks, 
                                                   dropout=None,
                                                   activation='relu',
                                                   last_block_final=False)

        # the last block is the one that will lead to final representation of the input
        self.representation_block = ExponentialClassifier(num_classes=10, 
                                                          in_features= 4 * self.emb_dim, 
                                                          num_layers=num_features_blocks, 
                                                          dropout=dropout, 
                                                          last_block_final=False)

        self.classification_head = nn.Linear(in_features=10, out_features=1)
        self.regression_head = nn.Linear(in_features=10, out_features=1)

        self.lr = learning_rate
        self.gamma = gamma

        self.save_hyperparameters()
        self.log_data = pd.DataFrame(data=[], columns=['image', 'predictions', 'labels', 'val_loss', 'epoch'])

    def forward(self, 
                x: torch.Tensor, 
                output: str = 'both') -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        if output not in ['both', 'classification', 'regression']: 
            raise ValueError(f"The 'output' argument is expected to belong to {['both', 'classification', 'regression']}\nFound: {output}")

        # make sure the input is 2d
        if x.ndim > 2: 
            raise ValueError(f"The input is expected to be 2 dimensional. Found: {x.shape}")
        # the first step is to extract the user and item ids        
        user_ids, item_ids = x[0, :], x[1, 0]
        # make sure user_ids and item_ids are of dim 1
        user_ids = torch.squeeze(user_ids) if user_ids.ndim != 1 else user_ids
        item_ids = torch.squeeze(item_ids) if item_ids.ndim != 1 else item_ids

        # pass the each set of ids to the corresponding embedding layer
        user_embds, item_embds = self.user_emb_layer.forward(user_ids), self.item_emb_layer.forward(item_ids)
        # pass the context to the context block
        context_features = self.context_block.forward(x[2:, :])

        # concatenate the embeddings and the context features
        representation = self.representation_block.forward(torch.cat([user_embds, item_embds, context_features], dim=1))

        clss, reg = self.classification_head.forward(representation), self.regression_head.forward(representation)

        if output == 'classification':
            return clss
        
        elif output == 'regression': 
            return reg
        
        return clss, reg

    def _forward_pass(self, batch, loss_reduced: bool = True):
        x, clss_y, reg_y = batch

        clss, reg = self.forward(x, output='both')

        # calculate the binary cross entropy loss
        cls_loss = F.binary_cross_entropy(clss, clss_y)
        # filter the regression output
        unseen_items_mask = clss_y != 0            
        reg = torch.masked_select(reg, unseen_items_mask)
        reg_y = torch.masked_select(reg_y, unseen_items_mask)
        # compute the mse loss
        reg_loss = F.mse_loss(reg, reg_y)

        return clss, reg, cls_loss, reg_loss            

    def training_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        _, _, cls_loss, reg_loss = self._forward_pass(batch)
        final_loss = reg_loss + self.coeff * cls_loss 

        self.log_dict({"train_cls_loss": cls_loss.cpu().item(), 
                       "train_reg_loss": reg_loss.cpu().item(), 
                       "loss": final_loss.cpu().item()})
        return final_loss 

    def validation_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> STEP_OUTPUT:        
        _, _, _, reg_loss = self._forward_pass(batch)
        # the batch is expected to have only positive samples    
        self.log_dict({'val_reg_loss': reg_loss.cpu().item()})

    def configure_optimizers(self):
        # since the encoder is pretrained, we would like to avoid significantly modifying its weights/
        # on the other hand, the rest of the AE should have higher learning rates.

        parameters = [{"params": self.avgpool.parameters(), "lr": self.lr},
                      {"params": self.head.parameters(), "lr": self.lr}]
        # add a learning rate scheduler        
        optimizer = optim.Adam(parameters, lr=self.lr)
        # create lr scheduler
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=self.gamma)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}



def train_classifier(configuration: Dict, 
                    train_csv_path: Union[str, Path], 
                    val_csv_path: Union[str, Path] = None,
                    log_dir: Union[str, Path] = None,
                    run_name: str = None,
                    batch_size: int = 32,
                    num_epochs: int = 10):

    wandb.init(project=WANDB_PROJECT_NAME, 
               name=run_name)
    
    wandb_logger = WandbLogger(project=WANDB_PROJECT_NAME,
                            log_model="all", 
                            save_dir=log_dir, 
                            name=run_name)

    # first process both directories
    train_csv_path = dirf.process_save_path(train_csv_path,
                                       file_ok=True,
                                       dir_ok=False
                                       )
    val_csv_path = dirf.process_save_path(val_csv_path, file_ok=True, dir_ok=False)

    # the output directory must be empty
    log_dir = os.path.join(SCRIPT_DIR, 'logs') if log_dir is None else log_dir
    # process the path
    log_dir = dirf.process_save_path(log_dir, file_ok=False)

    # create the dataset
    train_ds = RecSysDataset(
                    ratings=train_csv_path, 
                    user_data_cols=USER_DATA_COLS,
                    item_data_cols=ITEM_DATA_COLS, 
                    negative_sampling=True,
                    )  
    
    train_dl = DataLoader(train_ds, 
                                  batch_size=batch_size, 
                                  num_workers=0,
                                  shuffle=True,
                                  pin_memory=True)
    
    if val_csv_path is not None:
        val_ds = RecSysDataset(ratings=val_csv_path,
                               user_data_cols=USER_DATA_COLS, 
                               item_data_cols=ITEM_DATA_COLS,
                               negative_sampling=False,
                               use_all_history=True)
        
        val_dl = DataLoader(val_ds, 
                                    batch_size=batch_size, 
                                    num_workers=0,
                                    shuffle=False,
                                    pin_memory=True)
    else:
        val_dl = None

    checkpnt_callback = ModelCheckpoint(dirpath=log_dir, 
                                        save_top_k=5, 
                                        monitor="val_loss",
                                        mode='min', 
                                        # save the checkpoint with the epoch and validation loss
                                        filename='classifier-{epoch:02d}-{val_loss:06f}')

    # define the trainer
    trainer = L.Trainer(
                        accelerator='gpu', 
                        devices=1,
                        logger=wandb_logger,
                        default_root_dir=log_dir,
                        
                        max_epochs=num_epochs,
                        check_val_every_n_epoch=3,

                        deterministic=True,
                        callbacks=[checkpnt_callback])

    model = RecSys(num_users=len(train_ds.all_user_ids), 
                   num_items=len(train_ds.all_item_ids), 
                   context_length=len(train_ds.user_cols) + 2 * len(train_ds.item_cols), 
                   **configuration)
    
    print("model defined, training started.")
    trainer.fit(model=model,
                train_dataloaders=train_dl,
                val_dataloaders=val_dl
                )


def main(configuration, 
         run_name: str, 
         num_epochs:int):
    print("main started !!")
    wandb.login(key='36259fe078be47d3ffd8f3b2628a4d773c6e1ce7')

    train_csv_path = os.path.join(DATA_FOLDER, 'prepared', 'u1_train.csv')
    val_csv_path = os.path.join(DATA_FOLDER, 'prepared', f'u1_test.csv')

    logs = os.path.join(SCRIPT_DIR, 'classifier_runs')
    os.makedirs(logs, exist_ok=True)

    train_classifier(
            configuration=configuration,
            train_csv_path=train_csv_path, 
            val_csv_path=val_csv_path,
            run_name=run_name,
            batch_size=32,
            log_dir=os.path.join(logs, f'exp_{len(os.listdir(logs)) + 1}'),     
            num_epochs=num_epochs)    



if __name__ == '__main__':
    configuration = {"emb_dim": 20,
                     "num_context_blocks": 2, 
                     "num_features_blocks": 4, 
                     } 
    main(configuration=configuration, 
         num_epochs=15, 
         run_name='recommendation_system_1')
    # sanity_check(run_name='hypertune_scene_classifier')

