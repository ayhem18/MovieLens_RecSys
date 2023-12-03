import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from typing import Union, Sequence, Tuple

# a couple of visualization functions 
def plot_histogram(x: Union[pd.Series, Sequence], 
                   x_label: str, 
                   y_label: str,
                   title: str, 
                   fig_size: Tuple[int, int]=None,
                   ax = None) -> None:
    if fig_size is not None:
        plt.figure(figsize=fig_size)


    if ax is not None:
        # An "interface" to matplotlib.axes.Axes.hist() method
        n, _, _ = ax.hist(x=x, bins='auto', color='#0504aa',
                                    alpha=0.7, rwidth=0.85)
        ax.grid(axis='y', alpha=0.75)
        ax.xlabel(x_label)
        ax.ylabel(y_label)
        ax.xticks(rotation=45)
        ax.title(title)
        maxfreq = n.max()
        ax.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

    n, _, _ = plt.hist(x=x, bins='auto', color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=45)
    plt.title(title)
    maxfreq = n.max()
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    
def bar_plot(x: Sequence, 
             y: Sequence, 
             x_label: str, 
             y_label: str, 
             title: str,
             fig_size: Tuple[int, int], 
             ax = None, 
             show:bool =True,
             xticks=None) -> None:
    plt.figure(figsize = fig_size)
    
    if ax is not None:
        # creating the bar plot
        ax.bar(x, y, color ='maroon', width = 0.4)
        ax.set_xlabel(x_label)
        ax.set_xticks(ticks=np.linspace(min(x), max(x), num=10), rotation=90)
        # ax.xticks(rotation=90)
        ax.set_ylable(y_label)
        ax.title(title)
        # ax.show()
        # plt.xticks()
    else:
        # creating the bar plot
        plt.bar(x, y, color ='maroon', width = 0.4)
    if xticks is not None:
        plt.xticks(ticks=xticks, rotation=90)
    else:
        plt.xticks(rotation=90)
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    # plt.show()
    if show: 
        plt.show()