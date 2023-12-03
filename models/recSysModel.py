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
import wandb
import pandas as pd
import numpy as np

random.seed(69)
torch.manual_seed(69)

from torch import optim
from pathlib import Path
from typing import Any, Union, Sequence, Tuple, Dict, List
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch import nn
from collections import defaultdict

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
import utilities.pytorch_utilities as pu

from models.classification_head import ExponentialClassifier
from models.recSysDataset import RecSysDataset, USER_DATA_COLS, ITEM_DATA_COLS, RecSysInferenceDataset


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
        super().__init__(*args, **kwargs)
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

    def forward(self, 
                x: torch.Tensor, 
                output: str = 'both') -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        if output not in ['both', 'classification', 'regression']: 
            raise ValueError(f"The 'output' argument is expected to belong to {['both', 'classification', 'regression']}\nFound: {output}")

        # make sure the input is 2d
        if x.ndim > 2: 
            raise ValueError(f"The input is expected to be 2 dimensional. Found: {x.shape}")
        
        # convert to the float data type
        x = x.to(torch.float).to(pu.get_module_device(self))
        # the first step is to extract the user and item ids        
        user_ids, item_ids = x[:, 0].to(torch.long), x[:, 1].to(torch.long)
        # make sure user_ids and item_ids are of dim 1
        user_ids = torch.squeeze(user_ids) if user_ids.ndim != 1 else user_ids
        item_ids = torch.squeeze(item_ids) if item_ids.ndim != 1 else item_ids

        # pass each set of ids to the corresponding embedding layer
        user_embds, item_embds = self.user_emb_layer.forward(user_ids), self.item_emb_layer.forward(item_ids)
        # pass the context to the context block
        context_features = self.context_block.forward(x[:, 2:])

        # concatenate the embeddings and the context features
        representation = self.representation_block.forward(torch.cat([user_embds, item_embds, context_features], dim=1))

        clss, reg = self.classification_head.forward(representation), self.regression_head.forward(representation)

        if output == 'classification':
            return clss
        
        elif output == 'regression': 
            return reg
        
        return clss, reg

    def _forward_pass(self, batch):
        x, clss_y, reg_y,  = batch

        # convert all the tensors in the batch to the float dtype 
        x = x.to(torch.float)
        clss_y = clss_y.to(torch.float)
        reg_y = reg_y.to(torch.float)

        # the outputs of the forward method are logits
        class_logits, reg_logits = self.forward(x, output='both')
        
        # compute the actual predictions: classes and ratings
        class_predictions, reg_predictions = torch.squeeze(F.sigmoid(class_logits)), 4 * torch.squeeze(F.sigmoid(reg_logits)) + 1

        # calculate the binary cross entropy loss
        cls_loss = F.binary_cross_entropy(class_predictions, clss_y)
        # filter the regression output
        unseen_items_mask = clss_y != 0            
        reg_predictions = torch.masked_select(reg_predictions, unseen_items_mask)
        reg_y = torch.masked_select(reg_y, unseen_items_mask)
        # compute the mse loss
        reg_loss = F.mse_loss(reg_predictions, reg_y)

        # calculate the accuracy
        accuracy = torch.mean(((class_predictions > 0.5).to(torch.long) == clss_y.to(torch.long)).to(torch.float))

        return class_predictions, reg_predictions, cls_loss, reg_loss, accuracy            

    def training_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        _, _, cls_loss, reg_loss, accuracy = self._forward_pass(batch)
        final_loss = reg_loss + self.coeff * cls_loss 

        self.log_dict({"train_cls_loss": cls_loss.cpu().item(), 
                       "train_reg_loss": reg_loss.cpu().item(), 
                       "train_loss": final_loss.cpu().item(), 
                       "train_accuracy": accuracy})
        return final_loss 

    def validation_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> STEP_OUTPUT:        
        _, _, cls_loss, reg_loss, val_accuracy = self._forward_pass(batch)
        # the batch is expected to have only positive samples    
        self.log_dict({"val_cls_loss": cls_loss.cpu().item(), 
                       "val_reg_loss": reg_loss.cpu().item(), 
                       "val_accuracy": val_accuracy}) 

    def configure_optimizers(self):
        # since the encoder is pretrained, we would like to avoid significantly modifying its weights/
        # on the other hand, the rest of the AE should have higher learning rates.
        parameters = [{"params": self.item_emb_layer.parameters(), "lr": self.lr},
        {"params": self.user_emb_layer.parameters(), "lr": self.lr},
        {"params": self.context_block.parameters(), "lr": self.lr}, 
        {"params": self.representation_block.parameters(), "lr": self.lr},
        {"params": self.classification_head.parameters(), "lr": self.lr},
        {"params": self.regression_head.parameters(), "lr": self.lr},
        ]

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

    all_user_ids = list(range(1, 944))
    all_item_ids = list(range(1, 1683))

    # create the dataset
    train_ds = RecSysDataset(
                    ratings=train_csv_path, 
                    user_data_cols=USER_DATA_COLS,
                    item_data_cols=ITEM_DATA_COLS,
                    all_items_ids=all_item_ids,
                    all_users_ids=all_user_ids, 
                    negative_sampling=True)
    
    train_dl = DataLoader(train_ds, 
                                  batch_size=batch_size,
                                  shuffle=True,
                                  pin_memory=True,
                                collate_fn=RecSysDataset.collate_fn)

    
    if val_csv_path is not None:
        val_ds = RecSysDataset(ratings=val_csv_path,
                               user_data_cols=USER_DATA_COLS, 
                               item_data_cols=ITEM_DATA_COLS,
                               negative_sampling=False,
                               all_items_ids=all_item_ids,
                               all_users_ids=all_user_ids, 
                               use_all_history=True)
        
        val_dl = DataLoader(val_ds, 
                            batch_size=batch_size,
                            shuffle=False,
                            pin_memory=True,
                            collate_fn=RecSysDataset.collate_fn)

    else:
        val_dl = None

    checkpnt_callback = ModelCheckpoint(dirpath=log_dir, 
                                        save_top_k=5, 
                                        monitor="val_reg_loss",
                                        mode='min', 
                                        # save the checkpoint with the epoch and validation loss
                                        filename='classifier-{epoch:02d}-{val_reg_loss:06f}')

    # define the trainer
    trainer = L.Trainer(
                        accelerator='gpu',
                        devices=1,
                        logger=wandb_logger,
                        default_root_dir=log_dir,
                        
                        max_epochs=num_epochs,
                        check_val_every_n_epoch=3,
                        log_every_n_steps=25,
                        deterministic=True,
                        callbacks=[checkpnt_callback])

    model = RecSys(num_users=len(all_user_ids) + 1, 
                   num_items=len(all_item_ids) + 1, 
                   context_length=len(train_ds.user_cols) + 2 * len(train_ds.item_cols), 
                   **configuration)
    
    print("model defined, training started.")
    trainer.fit(model=model,
                train_dataloaders=train_dl,
                val_dataloaders=val_dl
                )


def main_train_function(configuration, 
         run_name: str, 
         num_epochs:int):
    print("main started !!")
    wandb.login(key='36259fe078be47d3ffd8f3b2628a4d773c6e1ce7')

    train_csv_path = os.path.join(DATA_FOLDER, 'prepared', 'u1_train.csv')
    val_csv_path = os.path.join(DATA_FOLDER, 'prepared', 'u1_test.csv')

    logs = os.path.join(SCRIPT_DIR, 'rs_runs')
    os.makedirs(logs, exist_ok=True)

    train_classifier(
            configuration=configuration,
            train_csv_path=train_csv_path, 
            val_csv_path=val_csv_path,
            run_name=run_name,
            batch_size=32,
            log_dir=os.path.join(logs, f'exp_{len(os.listdir(logs)) + 1}'),     
            num_epochs=num_epochs)    


def recommend(model: RecSys, 
              train_ratings: Union[pd.DataFrame, str, Path], 
              test_ratings: Union[pd.DataFrame, str, Path],
              all_users: List[int], 
              all_items: List[int],
              save_path: Union[str, Path],
              method: str = 'classification',  
              top_k_items: int = 20, 
              batch_size: int = 100) -> List[int]:
    
    metds = ['classification', 'regression', 'embeddings']
    if method not in metds:
        raise ValueError(f"The method is expected to be one of the following: {metds}. Found: {method}")

    # make sure to set the model to the evaluation state
    model.eval()

    train_ratings = pd.DataFrame(train_ratings) if isinstance(train_ratings, (Path, str)) else train_ratings
    test_ratings = pd.DataFrame(test_ratings) if isinstance(test_ratings, (Path, str)) else test_ratings

    # build an inference dataset
    inf_ds = RecSysInferenceDataset(train_ratings=train_ratings, 
                                    test_ratings=test_ratings,
                                    all_users_ids=all_users, 
                                    all_items_ids=all_items, 
                                    user_cols=USER_DATA_COLS, 
                                    item_cols=ITEM_DATA_COLS, 
                                    batch_size=batch_size)
    results_classification = defaultdict(lambda : {})
    results_regression = defaultdict(lambda: {})

    all_users = set(all_users)
    all_items = set(all_items)


    for user_id, model_input in inf_ds:
        # if user_id > 10:
        #     break

        item_ids = model_input[:, 1].detach().cpu().numpy().astype(np.int32)
        # get the model's output

        if user_id not in all_users:
            raise ValueError(f"check user_id. user_id: {user_id}")

        for iid in item_ids:
            if iid not in all_items:
                raise ValueError(f"check item id: {iid}")

        with torch.no_grad():
            class_logits, reg_logits = model.forward(model_input, output='both')
            class_predictions, reg_predictions = (torch.squeeze(F.sigmoid(class_logits)).detach().cpu().numpy(), 
                                                4 * torch.squeeze(F.sigmoid(reg_logits)).detach().cpu().numpy() + 1)
        
        if class_predictions.size == 1:
            if len(item_ids) != 1:
                raise ValueError(f"The number of predictions do not match the number of items")
            class_predictions = np.asarray([class_predictions.item()])

        elif len(class_predictions) != len(item_ids):
            raise ValueError(f"The number of predictions do not match the number of items. preds: {len(class_predictions)}, items: {len(item_ids)}")

        if reg_predictions.size == 1:
            if len(item_ids) != 1:
                raise ValueError(f"The number of predictions do not match the number of items")
            reg_predictions = np.asarray([reg_predictions.item()])

        elif len(reg_predictions) != len(item_ids):
            raise ValueError(f"The number of predictions do not match the number of items. preds: {len(reg_predictions)}, items: {len(item_ids)}")

        results_classification[user_id].update({i_id: cp for i_id, cp in zip(item_ids, class_predictions)})
        results_regression[user_id].update({i_id: cp for i_id, cp in zip(item_ids, reg_predictions)})


    for user_id, _ in results_classification.items():    
        # sort the results in descending order
        results_classification[user_id] = sorted(results_classification[user_id].items(), key=lambda x: x[1], reverse=True)
    
    for user_id, _ in results_regression.items():
        results_regression[user_id] = sorted(results_regression[user_id].items(), key=lambda x: x[1], reverse=True)

    if method == 'classification':
        recs = {user_id: [x[0] for x in res[:top_k_items]] for user_id, res in results_classification.items()}

    elif method == 'regression':
        recs = {user_id: [x[0] for x in res[:top_k_items]] for user_id, res in results_regression.items()}
    
    recs_csv = pd.DataFrame(data=recs).T
    recs_csv.columns = [f'item_{i}' for i in range(1, top_k_items + 1)] 

    recs_csv.to_csv(save_path)
    return recs


if __name__ == '__main__':
    configuration = {"emb_dim": 16,
                     "num_context_blocks": 4, 
                     "num_features_blocks":8, 
                     } 
    # main_train_function(configuration=configuration, 
    #      num_epochs=1000, 
    #      run_name='rs_bigger_scale')
    train_csv = pd.read_csv(os.path.join(DATA_FOLDER, 'prepared', 'u1_train.csv'))
    test_csv = pd.read_csv(os.path.join(DATA_FOLDER, 'prepared', 'u1_test.csv'))

    chkpnt_path = os.path.join(PARENT_DIR, 'models/rs_runs/exp_7/classifier-epoch=554-val_reg_loss=1.078150.ckpt')
    rec_sys = RecSys.load_from_checkpoint(checkpoint_path=chkpnt_path).to('cuda' if torch.cuda.is_available else 'cpu')

    # let's try recommending items
    recommendataions = recommend(model=rec_sys, 
                train_ratings=train_csv,
                test_ratings=test_csv, 
                all_users=list(range(1, 944)),
                all_items=list(range(1, 1683)),
                save_path=os.path.join(SCRIPT_DIR,'recommendations_classification.csv'),
                batch_size = 200
              )

    recommendataions = recommend(model=rec_sys, 
                train_ratings=train_csv,
                test_ratings=test_csv, 
                all_users=list(range(1, 944)),
                all_items=list(range(1, 1683)),
                save_path=os.path.join(SCRIPT_DIR,'recommendataions_regression.csv'),
                batch_size = 200,
                method='regression'
              )
    
    print(recommendataions)
