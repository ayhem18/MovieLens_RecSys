"""
This script contains the definition of the classifier based on top of the autoencoder 
"""

import os
import sys
import pytorch_lightning as L

from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
current = SCRIPT_DIR

while 'models' not in os.listdir(current):
    current = Path(current).parent

PARENT_DIR = str(current)
sys.path.append(str(current))
DATA_FOLDER = os.path.join(current, 'data')
print(DATA_FOLDER)

from models.recSysModel import main_train_function


if __name__ == '__main__':

    configuration = {"emb_dim": 16,
                     "num_context_blocks": 4, 
                     "num_features_blocks":8, 
                     } 
    
    main_train_function(configuration=configuration, 
         num_epochs=1000, 
         run_name='rs_bigger_scale')

    