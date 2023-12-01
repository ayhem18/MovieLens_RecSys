import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from typing import Union, Sequence, Tuple

# a couple of visualization functions 
def plot_histogram(x: Union[pd.Series, Sequence], 
                   x_label: str, 
                   y_label: str,
                   title: str, 
                   fig_size: Tuple[int, int]=None) -> None:
    
    if fig_size is not None:
        plt.figure(figsize=fig_size)
    # An "interface" to matplotlib.axes.Axes.hist() method
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
             fig_size: Tuple[int, int]) -> None:
    fig = plt.figure(figsize = fig_size)
    # creating the bar plot
    plt.bar(x, y, color ='maroon', width = 0.4)
    plt.xticks(rotation=90)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()