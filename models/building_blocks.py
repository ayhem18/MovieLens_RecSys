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


class LinearNet(nn.Module):
    def build(self,
              in_features: int,
              out_features: int,
              num_layers: int,
              activation: str,
              last_layer_is_final: bool = True):
        # the first and final layer are always present
        num_layers = num_layers + 2
        # compute the number of hidden units
        hidden_units = [int(u) for u in np.linspace(in_features, out_features, num=num_layers)]

        layers = [LinearBlock(in_features=hidden_units[i], out_features=hidden_units[i + 1], activation=activation,
                              is_final=False) for i in range(len(hidden_units) - 2)]

        # the last linear block should be set as 'final'
        layers.append(
            LinearBlock(in_features=hidden_units[-2], out_features=hidden_units[-1], is_final=last_layer_is_final,
                        activation=activation))

        return nn.Sequential(*layers)

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 num_layers: int = 1,
                 activation: str = 'leaky_relu',
                 last_layer_is_final: bool = True,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        # let's make it a simple encoder
        self.net = self.build(in_features=in_features,
                              out_features=out_features,
                              num_layers=num_layers,
                              activation=activation,
                              last_layer_is_final=last_layer_is_final)

    def forward(self, x: torch.Tensor):
        return self.net.forward(x)

    def __str__(self) -> str:
        return self.net.__str__()

    def children(self) -> Iterator[Module]:
        return self.net.children()

    def named_children(self) -> Iterator[Module]:
        return self.net.children()
