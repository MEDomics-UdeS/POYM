"""
Filename: visualization.py

Authors: Nicolas Raymond
         Hakima Laribi

Description: This file contains all function related to data visualization
"""

from os.path import join
from typing import List, Optional, Union, Dict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
from numpy import array
from numpy import sum as npsum
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch import tensor
import json
from src.data.processing.sampling import MaskType

# Epochs progression figure name
EPOCHS_PROGRESSION_FIG: str = "epochs_progression.png"


def format_to_percentage(pct: float, values: List[float]) -> str:
    """
    Change a float to a str representing a percentage
    Args:
        pct: count related to a class
        values: count of items in each class

    Returns: str
    """
    absolute = int(round(pct/100.*npsum(values)))
    return "{:.1f}%".format(pct, absolute)


def visualize_class_distribution(targets: array,
                                 label_names: dict,
                                 title: Optional[str] = None) -> None:
    """
    Shows a pie chart with classes distribution

    Args:
        targets: array of class targets
        label_names: dictionary with names associated to target values
        title: title for the plot

    Returns: None
    """

    # We first count the number of instances of each value in the targets vector
    label_counts = {v: npsum(targets == k) for k, v in label_names.items()}

    # We prepare a list of string to use as plot labels
    labels = [f"{k} ({v})" for k, v in label_counts.items()]

    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    fig, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie(label_counts.values(),
                                      textprops=dict(color="w"),
                                      startangle=90,
                                      autopct=lambda pct: format_to_percentage(pct, list(label_counts.values())))
    ax.legend(wedges, labels,
              title="Labels",
              loc="center right",
              bbox_to_anchor=(0.1, 0.5, 0, 0),
              prop={"size": 8})

    plt.setp(autotexts, size=8, weight="bold")

    if title is not None:
        ax.set_title(title)

    fig.tight_layout()
    plt.show()


def visualize_embeddings(embeddings: tensor,
                         category_levels: tensor,
                         continuous_levels: tensor = None,
                         perplexity: int = 10,
                         title: Optional[str] = None,
                         saving_path: Optional[str] = None) -> None:
    """
    Visualizes embeddings in a 2D space

    Args:
        embeddings: (N,D) tensor
        category_levels: (N,) tensor (with category indices)
        perplexity: perplexity parameter of TSNE
        title: title of the plot

    Returns: None
    """
    # Convert tensor to numpy array
    X = embeddings.numpy()
    y = category_levels.numpy()
    p = continuous_levels.numpy() if continuous_levels is not None else None

    # If the embeddings have more than 2 dimensions, project them with TSNE
    if X.shape[1] > 2:
        X = TSNE(n_components=2, perplexity=perplexity, random_state=101).fit_transform(X)

    # Create the plot
    for m in np.unique(y):
        x_m = np.where(y==m)
        label = 'Mortalité' if m==1 else 'Survie'
        marker = 'o' if m==1 else '*'
        color = 'crimson' if m==1 else 'limegreen'
        if p is None:
            plt.scatter(X[x_m, 0], X[x_m, 1], c= color,  label=label, marker=marker)
        else:
            plt.scatter(X[x_m, 0], X[x_m, 1], c=p[x_m], label=label, marker=marker)

    if title is not None:
        plt.title(title)
    else:
        plt.title('Embeddings visualization with TSNE')

    if p is not None:
        plt.colorbar()

    plt.legend()

    if saving_path is not None:
        plt.savefig(saving_path)

    plt.show()
    plt.close()


def visualize_epoch_progression(train_history: List[tensor],
                                valid_history: List[tensor],
                                progression_type: List[str],
                                path: str) -> None:
    """
    Visualizes train and test loss histories over evaluating epoch

    Args:
        train_history: list of (E,) tensors where E is the number of epochs
        valid_history: list of (E,) tensor
        progression_type: list of string specifying the type of the progressions to visualize
        path: path where to save the plots

    Returns: None
    """
    mpl.use('TkAgg')
    plt.figure(figsize=(12, 8))

    # If there is only one plot to show (related to the loss)
    if len(train_history) == 1:

        x = range(len(train_history[0]))
        plt.plot(x, train_history[0], label=MaskType.TRAIN)
        plt.plot(x, valid_history[0], label=MaskType.VALID)

        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel(progression_type[0])

    # If there are two plots to show (one for the loss and one for the evaluation metric)
    else:
        for i in range(len(train_history)):

            nb_epochs = len(train_history[i])
            plt.subplot(1, 2, i+1)
            plt.plot(range(nb_epochs), train_history[i], label=MaskType.TRAIN)
            if len(valid_history[i]) != 0:
                plt.plot(range(nb_epochs), valid_history[i], label=MaskType.VALID)

            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel(progression_type[i])

    plt.tight_layout()
    plt.savefig(join(path, EPOCHS_PROGRESSION_FIG))
    plt.close()


def pca_visualization_2D(x: Union[array, tensor],
                         y: Union[array, tensor],
                         path: str,
                         figure_name: str,
                         colors: list) -> None:
    """
    Visualizes datas in a 2D plan using PCA

    Args:
        x: data
        y: ground truth
        path: saving path
        figure_name: name of the generated figure
        colors: colors to associate to each class

    Returns: None
    """

    # Launch the principal component analysis
    pca = PCA(n_components=2).fit_transform(x)
    x_pca = pd.DataFrame(data=pca, columns=['principal component 1', 'principal component 2'])

    # Get targets
    targets = np.unique(y)

    # Check colors validity
    if len(colors) < len(targets):
        raise ValueError(f"{len(colors)} < {len(targets)}. Each class must have a color associated to.")

    else:
        colors = colors[0:len(targets)-1]

    # Plot the figure
    plt.figure()
    plt.figure(figsize=(10, 10))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    plt.xlabel('Principal Component - 1', fontsize=20)
    plt.ylabel('Principal Component - 2', fontsize=20)
    plt.title("Principal Component Analysis", fontsize=20)

    for target, color in zip(targets, colors):
        indices = (y == target).nonzero(as_tuple=True)
        plt.scatter(x_pca.loc[indices, 'principal component 1'], x_pca.loc[indices, 'principal component 2'],
                    c=color, s=50)

    plt.legend(targets, prop={'size': 15})
    plt.savefig(join(path, figure_name))

    # clear figure
    plt.clf()
    # close figure
    plt.close()


def pca_visualization_3D(x: Union[array, tensor],
                         y: Union[array, tensor],
                         path: str,
                         figure_name: str,
                         colors: Dict[int, str] = None,
                         labels: Dict[int, str] = None,
                         marker: Dict[int, str] = None,
                         sizes: Dict[int, str] = None) -> None:
    """
    Visualizes datas in a 3D plan using PCA

    Args:
        x: data
        y: ground truth
        path: saving path
        figure_name: name of the generated figure
        colors: colors to associate to each class
        labels: labels to associate to each set of classes
        marker: markers to associate to each set of classes
        sizes: sizes to associate to each marker

    Returns: None
    """

    # Launch the principal component analysis
    x_pca = PCA(n_components=2).fit_transform(x)

    # Get targets
    targets = np.unique(y)

    # Validation of inputs
    if len(colors.keys()) != len(targets) or len(labels.keys()) != len(targets) or len(marker.keys()) != len(targets) \
            or len(sizes.keys()) != len(targets):
        raise ValueError('Inputs should have same size as number of classes')

    # Variance obtained in each axe
    ex_variance = np.var(x_pca, axis=0)
    ex_variance_ratio = ex_variance / np.sum(ex_variance)

    # Get data values of each axe
    x_ax = x_pca[:, 0]
    y_ax = x_pca[:, 1]
    z_ax = x_pca[:, 2]

    # Plot the figure
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('white')
    for l in np.unique(y):
        ix = np.where(y == l)
        ax.scatter(x_ax[ix], y_ax[ix], z_ax[ix], c=colors[l], s=40,
                   label=labels[l], marker=marker[l], alpha=sizes[l])

        # for loop ends
    ax.set_xlabel(f"1st Principal Component. Variance: {ex_variance_ratio[0]}", fontsize=14)
    ax.set_ylabel(f"2nd Principal Component. Variance: {ex_variance_ratio[1]}", fontsize=14)
    ax.set_zlabel(f"3rd Principal Component. Variance: {ex_variance_ratio[2]}", fontsize=14)

    ax.legend()

    plt.savefig(join(path, figure_name))
    # clear figure
    plt.clf()
    # close figure
    plt.close()


def categories_proportion(x: Union[array, tensor],
                          path: str,
                          figure_name: str
                          ) -> None:
    """
    Visualizes a column categories proportions in a pie chart

    Args:
        x: column data
        path: saving path
        figure_name: name of the generated figure

    Returns: None
    """
    plt.figure(figsize=(12, 8))

    categories = np.unique(x)
    proportions = [npsum(np.where(x == category)) / len(x) * 100 for category in categories]

    plt.pie(proportions, labels=categories, autopct='%1.1f%%', shadow=False, startangle=90)

    # save the figure
    plt.savefig(join(path, figure_name))
    # clear figure
    plt.clf()
    # close figure
    plt.close()


def continuous_columns_distribution(x: Union[array, tensor],
                                    path: str,
                                    figure_name: str
                                    ) -> None:
    """
    Visualizes a column continuous distribution in a histogram

    Args:
        x: column data
        path: saving path
        figure_name: name of the generated figure

    Returns: None
    """
    plt.style.use('ggplot')
    plt.hist(x, bins=np.max(x) - np.min(x))

    plt.ylabel('frequency')

    # save the figure
    plt.savefig(join(path, figure_name))
    # clear figure
    plt.clf()
    # close figure
    plt.close()











