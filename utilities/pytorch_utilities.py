"""
Unlike helper_functionalities.py, this script contains, Pytorch code that is generally used across different
scripts and Deep Learning functionalities
"""

import gc
import torch
from torch import nn
from typing import Union
from torch.utils.data import DataLoader
from pathlib import Path
import os
from datetime import datetime as d
from utilities.directories_and_files import process_save_path

HOME = os.getcwd()


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# set the default device
def get_default_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def get_module_device(module: nn.Module) -> str:
    # this function is mainly inspired by this overflow post:
    # https://stackoverflow.com/questions/58926054/how-to-get-the-device-type-of-a-pytorch-module-conveniently
    if hasattr(module, 'device'):
        return module.device
    return next(module.parameters()).device


def __verify_extension(p):
    return os.path.basename(p).endswith('.pt') or os.path.basename(p).endswith('.pth')


def save_model(model: nn.Module, path: Union[str, Path] = None) -> None:
    # the time of saving the model
    now = d.now()
    file_name = "-".join([str(now.month), str(now.day), str(now.hour), str(now.minute)])
    # add the extension
    file_name += '.pt'

    # first check if the path variable is None:
    path = path if path is not None else os.path.join(HOME, file_name)

    # process the path
    path = process_save_path(path,
                             dir_ok=True,
                             file_ok=True,
                             condition=lambda p: not os.path.isfile(p) or __verify_extension(p),
                             error_message='MAKE SURE THE FILE PASSED IS OF THE CORRECT EXTENSION')

    if os.path.isdir(path):
        path = os.path.join(path, file_name)

    # finally save the model.
    torch.save(model.state_dict(), path)


def load_model(base_model: nn.Module,
               path: Union[str, Path]) -> nn.Module:
    # first process the path
    path = process_save_path(path,
                             dir_ok=False,
                             file_ok=True,
                             condition=lambda p: not os.path.isfile(p) or __verify_extension(p),
                             error_message='MAKE SURE THE FILE PASSED IS OF THE CORRECT EXTENSION')

    base_model.load_state_dict(torch.load(path))

    return base_model
