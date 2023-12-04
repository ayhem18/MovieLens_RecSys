# first let's build a simple linear block
import torch
import numpy as np
import random
import itertools

from torch import nn
from typing import Iterator, Tuple, Union, List

import os, sys
from pathlib import Path
from abc import ABC, abstractmethod

from torch.nn.modules.module import Module

# let's first set the random seeds 
random.seed(69)
np.random.seed(69)
torch.manual_seed(69)


class LinearBlock(nn.Module):
    """A linear block that adds both activation, dropout and BatchNormalization on top of
    a nn.Linear pytorch layer
    """
    # let's define a class method that will map the activation name to the correct layer
    activation_functions = {"leaky_relu": nn.LeakyReLU,
                            "relu": nn.ReLU,
                            "tanh": nn.Tanh}

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 activation: str = 'leaky_relu',
                 is_final: bool = True,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # the idea here is quite simple. 
        components = [nn.Linear(in_features=in_features, out_features=out_features)]
        # depending on the value of 'is_final' 
        if not is_final:
            norm_layer = nn.BatchNorm1d(num_features=out_features)
            activation_layer = self.activation_functions[activation]()
            components.extend([norm_layer, activation_layer])

        self._block = nn.Sequential(*components)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._block.forward(x)

    def __str__(self) -> str:
        return self._block.__str__()

    @property
    def block(self):
        return self._block

    @block.setter
    def block(self, new_block: Union[nn.Sequential, nn.Module]):
        # make sure the new block is either 
        if isinstance(new_block, (nn.Sequential, nn.Module)):
            raise TypeError((f"The block is expected to be either of type {nn.Module} or {nn.Sequential}\n"
                             f"Found: {type(new_block)}"))

    def children(self) -> Iterator[Module]:
        return self.block.children()

    def named_children(self) -> Iterator[Module]:
        return self.block.children()