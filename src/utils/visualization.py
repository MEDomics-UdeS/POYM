"""
Filename: visualization.py

Authors: Nicolas Raymond
         Hakima Laribi

Description: This file contains all function related to data visualization
"""
import os
from os.path import join
from re import search
from typing import List, Optional, Union, Dict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
from numpy import array
from numpy import sum as npsum
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch import tensor
import seaborn as sns
import json
from src.data.processing.sampling import MaskType
import textwrap

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
    absolute = int(round(pct / 100. * npsum(values)))
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
        x_m = np.where(y == m)
        label = 'Mortalité' if m == 1 else 'Survie'
        marker = 'o' if m == 1 else '*'
        color = 'crimson' if m == 1 else 'limegreen'
        if p is None:
            plt.scatter(X[x_m, 0], X[x_m, 1], c=color, label=label, marker=marker)
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
        if len(valid_history) > 0:
            plt.plot(x, valid_history[0], label=MaskType.VALID)

        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel(progression_type[0])

    # If there are two plots to show (one for the loss and one for the evaluation metric)
    else:
        for i in range(len(train_history)):

            nb_epochs = len(train_history[i])
            plt.subplot(1, 2, i + 1)
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
        colors = colors[0:len(targets) - 1]

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


def plot_heatmaps_ELSTM(lstm_aucs: pd.DataFrame,
                        strategy: str = 'exact',
                        image_label: str = 'heatmap'):
    """
    Plots a heatmap with the AUCs of each LSTM constituting the ELSTM on each group of
    patients from the test set

    Args:
        lstm_aucs: AUCs of each LSTM constituting the ELSTM
        strategy:
                if 'exact', compare the AUCs on the last visits of patients
                if 'with', compare the AUCs on the intermediate visits of patients
        image_label: label of the resulting image
    """
    mpl.use('TkAgg')
    mpl.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    suffix = '' if strategy != 'exact' else ',last'
    last_visit = 'Last\nvisit' if strategy == 'exact' else 'Any\nvisit'
    y_labels = [f'$V_{{{str(i) + suffix}}}$' for i in range(1, lstm_aucs.shape[0] - 1)] + [f'$V_{{>5{suffix}}}$'] + [
        last_visit]
    x_labels = [f'$LSTM_{{{i}}}$' for i in range(1, lstm_aucs.shape[1])] + ['$LSTM_{last}$']
    lstm_aucs = lstm_aucs.astype(float)

    # Sets the minimum and maximum of the color bar scale
    scale_min, scale_max = 60., 93.
    scale_med = (scale_max + scale_min) / 2

    mpl.use('TkAgg')

    # Create a figure and subplots
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)

    # Plot the second heatmap on ax2 with common vmin and vmax
    heatmap = ax.imshow(lstm_aucs, cmap='BuPu', vmin=scale_min, vmax=scale_max, aspect='auto')
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, fontsize=30)
    ax.set_yticklabels(y_labels, fontsize=30)
    ax.xaxis.set_ticks_position('top')

    for i in range(lstm_aucs.shape[0]):
        for j in range(lstm_aucs.shape[1]):
            if lstm_aucs.iloc[i, j] != None:
                font_color = 'black' if lstm_aucs.iloc[i, j] < scale_med else 'white'
                ax.text(j, i, f'{lstm_aucs.iloc[i, j]:.1f}', ha='center', va='center', color=font_color,
                        fontsize=21,
                        fontweight='bold')

    plt.savefig(f'{image_label}.svg')


def time_feature_importance(dfs: List[pd.DataFrame], figure_name: str):
    """
    Plots the feature importance of each set of predictors (Demographics, admission characteristics,
    comorbidity diagnoses, admission diagnoses) from previous and current visits for each groups of patients

    Args:
        dfs: List of dataframes containing AUC scores before and after swapping each predictor in the dataset
        for each group of patients
        figure_name: label of the resulting image
    """
    feature_importance_global = {
        'Demographics': {type_diff: {nb_visits: [] for nb_visits in range(len(dfs))} for type_diff in
                         ['current', 'previous']},
        'Admission characteristics': {type_diff: {nb_visits: [] for nb_visits in range(len(dfs))} for type_diff in
                                      ['current', 'previous']},
        'Comorbidity diagnoses': {type_diff: {nb_visits: [] for nb_visits in range(len(dfs))} for type_diff in
                                  ['current', 'previous']},
        'Admission diagnoses': {type_diff: {nb_visits: [] for nb_visits in range(len(dfs))} for type_diff in
                                ['current', 'previous']},
    }
    single_feature_importance_global = {
        feature: {type_diff: {nb_visits: [] for nb_visits in range(len(dfs))} for type_diff in
                  ['current', 'previous']} for feature in dfs[0].loc[:, 'Features']
    }
    # Compute importance
    for nb_visits, df in enumerate(dfs):

        for i in range(1, 101):
            feature_importance = {
                'Demographics': {'current': 0, 'previous': 0},
                'Admission characteristics': {'current': 0, 'previous': 0},
                'Comorbidity diagnoses': {'current': 0, 'previous': 0},
                'Admission diagnoses': {'current': 0, 'previous': 0}
            }
            single_feature_importance = {feature: {'current': 0, 'previous': 0} for feature in
                                         dfs[0].loc[:, 'Features']}
            if not df.loc[:, f'AUC_BASE_{i}'].isna().any():
                sumratio = 0
                diff = {}
                for type_diff in ['current', 'previous']:
                    diff[type_diff] = df.loc[:, f'AUC_BASE_{i}'] - df.loc[:, f'AUC_{type_diff}_{i}']

                for feature_index in range(df.shape[0]):
                    feature = df.iloc[feature_index].loc['Features']
                    if search("^(age.*)|^(gender.*)", feature):
                        group_of_features = 'Demographics'
                    elif search("^(adm_.*)", feature):
                        group_of_features = 'Admission diagnoses'
                    elif search("^(dx_.*)|(has_dx)", feature):
                        group_of_features = 'Comorbidity diagnoses'
                    else:
                        group_of_features = 'Admission characteristics'

                    for type_diff in ['current', 'previous']:
                        ratio = diff[type_diff].iloc[feature_index] if diff[type_diff].iloc[feature_index] > 0 else 0
                        feature_importance[group_of_features][type_diff] += ratio
                        single_feature_importance[feature][type_diff] = ratio
                        sumratio += ratio

                if sumratio != 0:
                    for group in feature_importance.keys():
                        for type_diff in ['current', 'previous']:
                            feature_importance_global[group][type_diff][nb_visits].append(
                                feature_importance[group][type_diff] / sumratio * 100)
                    for feature in single_feature_importance.keys():
                        for type_diff in ['current', 'previous']:
                            single_feature_importance_global[feature][type_diff][nb_visits].append(
                                single_feature_importance[feature][type_diff] / sumratio * 100)
                else:
                    print('iw')

    mpl.use('TkAgg')
    sns.set(style="whitegrid")  # Set seaborn style to whitegrid
    # Plot feature importance as curves
    mpl.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    fig, ax = plt.subplots(figsize=(18, 10))  # Adjust the figure size as needed

    # Specify a custom pastel color palette
    pastel_palette2 = sns.color_palette("pastel", len(feature_importance_global))
    pastel_palette = ['#FDDD5C'] + [pastel_palette2[2]] + [pastel_palette2[3]] + [pastel_palette2[0]]

    # Set the font size for the axis labels and tick labels
    ax.tick_params(axis='both', which='major', labelsize=42)
    ax.set_ylabel('% Feature importance', fontsize=42, labelpad=20)
    pt_visits = ['All'] + [f'$V_{{{i},last}}$' for i in range(1, len(dfs) - 1)] + [f'$V_{{>{len(dfs) - 2},last}}$']

    # Loop through the feature importances and plot curves
    for (group, dict_type), color in zip(feature_importance_global.items(), pastel_palette):

        for type_features, aucs_boot_per_visit in dict_type.items():
            ls = '--' if type_features == 'previous' else '-'
            values = [np.mean(feature_imp) for feature_imp in aucs_boot_per_visit.values()]
            yerr = [np.std(feature_imp) for feature_imp in aucs_boot_per_visit.values()]

            ax.plot(pt_visits, values, color=color, linewidth=3.5, markersize=11, marker='D', ls=ls, )

            # Add shaded error regions
            ax.fill_between(pt_visits, np.array(values) - np.array(yerr), np.array(values) + np.array(yerr),
                            color=color, alpha=0.3, linewidth=2.)

    ax.grid(True, linestyle='--', alpha=0.7, linewidth=1.2)

    for ft_type, ls in zip(['Previous features', 'Current features'], ['--', '-']):
        ax.plot(np.NaN, np.NaN, ls=ls,
                label=ft_type, c='black', linewidth=3.5, markersize=11, marker='D', zorder=0)

    ax2 = ax.twinx()

    for group, color in zip(feature_importance_global.keys(), pastel_palette):
        ax2.plot(np.NaN, np.NaN, c=color, label=group, linewidth=3.5, markersize=11, marker='D', zorder=5)

    ax2.get_yaxis().set_visible(False)
    ax2.get_xaxis().set_visible(False)
    # Customize the legend
    ax2.legend(fontsize=35, loc='upper left', bbox_to_anchor=(0.45, 1.3), framealpha=1.)
    ax.legend(fontsize=35, loc='upper left', bbox_to_anchor=(0., 1.3), framealpha=None)

    plt.tight_layout()
    plt.savefig(figure_name)
    plt.clf()


def proportion_of_survival_mortality(df, nb_visits, figure_name):
    """
    Plots the proportion of survival and mortality for each group of patients

    Args:
        df: HOMR dataset
        nb_visits: number of patients groups to consider
        figure_name: label of the resulting image
    """
    mpl.use('TkAgg')
    mpl.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    # Find the index of the row with the highest 'nb_visit' for each patient
    idx_max_visits = df.groupby('patient_id')['nb_visits'].idxmax()

    # Use the index to extract the corresponding rows
    max_visits_rows = df.loc[idx_max_visits, ['patient_id', 'oym']]

    # Get the number of visits for each patient
    visits_count = df.groupby('patient_id').size().reset_index(name='visits_count')

    # Merge the 'visits_count' with the rows containing the highest 'nb_visit' for each patient
    grouped_patients = pd.merge(max_visits_rows, visits_count, on='patient_id')

    task_counts = grouped_patients.groupby(['visits_count', 'oym']).size().reset_index(
        name='patient_count')

    n_visits = ['all'] + [i for i in range(1, nb_visits + 1)] + ['>5']
    pt_visits = ['All'] + [f'$V_{{{i},last}}$' for i in range(1, nb_visits + 1)] + [f'$V_{{>{nb_visits},last}}$']
    col = ['Survival', 'Mortality']
    bar_visits = pd.DataFrame(index=pt_visits, columns=col)
    all, mor_oym = [], []
    for v in n_visits:
        if v == n_visits[-1]:
            mor_oym.append(
                task_counts[(task_counts['visits_count'] > nb_visits) & (task_counts['oym'] == True)][
                    'patient_count'].sum())
            all.append(task_counts[(task_counts['visits_count'] > nb_visits)]['patient_count'].sum())

        elif v == n_visits[0]:
            mor_oym.append(task_counts[(task_counts['oym'] == True)]['patient_count'].sum())
            all.append(task_counts['patient_count'].sum())

        else:
            mor_oym.append(
                task_counts[(task_counts['visits_count'] == v) & (task_counts['oym'] == True)]['patient_count'].sum())
            all.append(task_counts[(task_counts['visits_count'] == v)]['patient_count'].sum())

    bar_visits.loc[pt_visits, col[0]] = np.array(all) - np.array(mor_oym)
    bar_visits.loc[pt_visits, col[1]] = mor_oym

    mpl.rcParams['figure.figsize'] = (15, 8)

    ax = bar_visits.iloc[1:, :].loc[:, col[:2]].plot.bar(stacked=False, align='center', width=0.7, zorder=2,
                                                         linewidth=1.2,
                                                         color=['#81CAD6', '#DC3E26'])
    plt.legend(fontsize=40)

    plt.ylabel('#Patients', fontsize=40, labelpad=20)
    plt.grid(True, linestyle='--', alpha=0.8, linewidth=1.2, zorder=0, axis='y')
    plt.xticks(fontsize=32, rotation=0)
    plt.yticks(fontsize=32)
    plt.tight_layout()
    plt.savefig(figure_name)
    plt.clf()


def time_between_visits(df, nb_visits, figure_name):
    """
    Plots the difference of months between the first visit discharge and the upcoming visits

    Args:
        df: HOMR dataset
        nb_visits: the rank of the visit up to which we compute the time difference
        figure_name: label of the resulting image
    """
    mpl.use('TkAgg')
    mpl.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'

    # Erase patients with only 1 visit and erase them from the dataset
    duplicate_patients = df[df.duplicated('patient_id', keep=False)]['patient_id']
    df = df[df['patient_id'].isin(duplicate_patients)]

    # Get delta time visits
    df['admission_date'] = pd.to_datetime(df['admission_date'])
    df['discharge_date'] = pd.to_datetime(df['discharge_date'])
    df = df.sort_values(by=['patient_id', 'admission_date'])
    df['first_discharge_date'] = df.groupby('patient_id')['discharge_date'].transform('first')
    df['time_between'] = (df['admission_date'] - df['first_discharge_date']).dt.days

    # Get the dictionary of differences between visits
    delta_visits = {str(n_visit): [value / 30 for value in df[df['nb_visits'] == n_visit]['time_between'].values]
                    for n_visit in range(2, nb_visits + 1)}
    delta_visits[f' > {nb_visits}'] = [value / 30 for value in df[df['nb_visits'] > nb_visits]['time_between'].values]

    mpl.rcParams['figure.figsize'] = (20, 11)
    pastel_palette2 = sns.color_palette("pastel", nb_visits)
    f, axes = plt.subplots(nb_visits, sharex=True)

    for a in range(nb_visits - 1):
        df_visits = pd.DataFrame()
        df_visits[str(a + 2)] = [value / 30 for value in df[df['nb_visits'] == a + 2]['time_between'].values]
        df_visits[f'oym'] = [value for value in df[df['nb_visits'] == a + 2]['oym'].values]
        key = list(delta_visits.keys())[a]
        sns.histplot(df_visits, x=df_visits[key], ax=axes[a], edgecolor='black', bins=30,
                     color=pastel_palette2[a], kde=True, hue='oym')
        axes[a].tick_params(axis='x', which='major', labelsize=30, length=5, width=3)
        axes[a].tick_params(axis='y', which='major', labelsize=18)
        axes[a].set_ylabel(f'$V_{{{key}}}$', fontsize=30)
        pos = np.percentile(axes[a].get_yticks(), 40)  # f a != 4 else 750
        axes[a].text(100, pos, f'n = {len(delta_visits[key])}', va='center', fontsize=30, fontweight='bold')

    f.text(0.01, 0.5, '# Visits', va='center', rotation='vertical', fontsize=50)
    plt.xlabel('Time (months)', fontsize=45)
    f.legend(fontsize=40, ncols=nb_visits, loc='upper center', bbox_to_anchor=(0.55, 1.02))
    f.subplots_adjust(left=0.12, right=0.99, top=0.9, bottom=0.1)
    plt.savefig(figure_name)
    plt.clf()


def plot_calibration_curves(files: List[str],
                            labels: List[str],
                            n_bootstraps=100,
                            figure_name='figures/calibration.svg'):
    """
    Plots the calibration curves over multiple bootstraps, uses interpolation unify the number of bins

    Args:
        files: files names with the predictions for each model
        labels: label of each model
        n_bootstraps: number of bootstraps
        figure_name: label of the resulting image
    """

    mpl.use('TkAgg')
    mpl.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    fig, ax = plt.subplots(figsize=(18, 10))  # Adjust the figure size as needed
    values = np.arange(start=0., stop=1., step=0.05)
    # Set the font size for the axis labels and tick labels
    ax.tick_params(axis='both', which='major', labelsize=42)
    ax.set_xlabel('Predicted risk', fontsize=42)
    ax.set_ylabel('Observed Y = 1', fontsize=42)
    np.random.seed(1)
    colors = ['#8DE5A1', '#FFBC82']
    for file_name, label, color in zip(files, labels, colors):
        y_axis = {'_random_visit': [], '': []}
        for i in range(n_bootstraps):
            with open(os.path.join(file_name, f'Split_{i}/records.json'), "r") as file:
                f = json.load(file)

            for section in ['', '_random_visit']:
                y_prob = [float(f[f'test{section}_results'][_id]['probability'])
                          for _id in f[f'test{section}_results'].keys()]
                y_true = [int(f[f'test{section}_results'][_id]['target'])
                          for _id in f[f'test{section}_results'].keys()]
                x, y = calibration_curve(y_true, y_prob, n_bins=10)
                y_axis[section].append(np.interp(values, y, x))

        for section in ['', '_random_visit']:
            ls = '--' if section == '_random_visit' else '-'
            data_array = np.array(y_axis[section])
            ymean = np.mean(data_array, axis=0)
            yerr = np.std(data_array, axis=0)
            ax.plot(values, ymean, color=color, linewidth=3.5, ls=ls)
            ax.fill_between(values, np.array(ymean) - np.array(yerr), np.array(ymean) + np.array(yerr),
                            color=color, alpha=0.3, linewidth=1.)

    ax.plot([0, 1], [0, 1], color='grey', linewidth=3.5, ls='dotted')
    ax.grid(True, linestyle='--', alpha=0.7, linewidth=1.2)
    for ft_type, ls in zip(['Any visit', 'Last visit'], ['--', '-']):
        ax.plot(np.NaN, np.NaN, ls=ls,
                label=ft_type, c='black', linewidth=3.5, zorder=0)
    ax2 = ax.twinx()

    for group, color in zip(labels + ['Perfectly calibrated'], colors + ['grey']):
        if color != 'grey':
            ax2.plot(np.NaN, np.NaN, c=color, label=group, linewidth=3.5, zorder=5)
        else:
            ax2.plot(np.NaN, np.NaN, c=color, label=group, linewidth=3.5, zorder=5,
                     ls='dotted')

    ax2.get_yaxis().set_visible(False)
    ax2.get_xaxis().set_visible(False)

    ax2.legend(fontsize=35, loc='upper left')
    ax.legend(fontsize=35, loc='upper center', bbox_to_anchor=(0.57, 1.))
    plt.tight_layout()
    plt.savefig(figure_name)
    plt.clf()


def plot_calibration_curves_for_bootstraps(files: List[str],
                                           labels: List[str],
                                           n_bootstraps=100,
                                           figure_name='figures/calibration_{}.svg'):
    """
    Plots the calibration curves for each bootstrap

    Args:
        files: files names with the predictions for each model
        labels: label of each model
        n_bootstraps: number of bootstraps
        figure_name: label of the resulting image
    """
    mpl.use('TkAgg')
    mpl.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    colors = ['#8DE5A1', '#FFBC82']

    for file_name, label, color in zip(files, labels, colors):
        for section in ['', '_random_visit']:
            fig, ax = plt.subplots(figsize=(18, 10))
            # Set the font size for the axis labels and tick labels
            ax.tick_params(axis='both', which='major', labelsize=42)
            ax.set_xlabel('Predicted risk', fontsize=42)
            ax.set_ylabel('Observed Y = 1', fontsize=42)

            ax.plot([0, 1], [0, 1], color='grey', linewidth=3.5, ls='dotted', label='Perfectly Calibrated')
            ax.grid(True, linestyle='--', alpha=0.7, linewidth=1.2)
            for i in range(n_bootstraps):
                with open(os.path.join(file_name, f'Split_{i}/records.json'), "r") as file:
                    f = json.load(file)

                y_prob = [float(f[f'test{section}_results'][_id]['probability'])
                          for _id in f[f'test{section}_results'].keys()]
                y_true = [int(f[f'test{section}_results'][_id]['target'])
                          for _id in f[f'test{section}_results'].keys()]
                x, y = calibration_curve(y_true, y_prob, n_bins=10)
                if i < n_bootstraps - 1:
                    ax.plot(y, x, color=color, linewidth=3.5)
                else:
                    section_type = 'Last visit' if section == '' else 'Any visit'
                    ax.plot(y, x, color=color, linewidth=3.5, label=f'{label} - {section_type}')

            ax.legend(fontsize=35)

            plt.savefig(figure_name.format(f'{label}{section}'), bbox_inches='tight')
            plt.clf()


def plot_alerts(files: List[str],
                labels: List[str],
                figure_name: str = 'alerts.svg'):
    """
    Plots the number of positive predictions, true positives and false positives for each model

    Args:
        files: files names with the predictions for each model
        labels: label of each model
        figure_name: label of the resulting image
    """
    mpl.use('TkAgg')
    mpl.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    fig, ax = plt.subplots(figsize=(18, 10))  # Adjust the figure size as needed

    # Set the font size for the axis labels and tick labels
    ax.tick_params(axis='both', which='major', labelsize=42)
    ax.set_ylabel('# Patients', fontsize=42)
    ax.set_xlabel('Positive predictions / CSOs', fontsize=42, labelpad=25)
    colors = sns.color_palette("pastel", len(files) + 2)
    np.random.seed(1)
    colors_dx = np.random.choice(len(colors), len(files), replace=False, )
    colors = [colors[i] for i in colors_dx]
    x_axis = ['All', 'FPs', 'TPs']
    for file_name, label, color in zip(files, labels, colors):
        with open(file_name, "r") as file:
            f = json.load(file)

        for section in ['', '_random_visit']:
            values = [float(f[f'test{section}_metrics'][metric]['mean']) for metric in
                      ['FalsePositives', 'TruePositives']]
            variation = [float(f[f'test{section}_metrics'][metric]['std']) for metric in
                         ['FalsePositives', 'TruePositives']]
            all_values = np.add(np.array(f[f'test{section}_metrics']['TruePositives']['values']),
                                np.array(f[f'test{section}_metrics']['FalsePositives']['values']))
            values.insert(0, np.mean(all_values))
            variation.insert(0, np.std(all_values))
            ls = '--' if section == '_random_visit' else '-'
            ax.plot(x_axis, values, color=color, linewidth=3.5, ls=ls, marker='D', markersize=11)
            ax.fill_between(x_axis, np.array(values) - np.array(variation), np.array(values) + np.array(variation),
                            color=color, alpha=0.3, linewidth=1.)

    ax.grid(True, linestyle='--', alpha=0.7, linewidth=1.2)
    for ft_type, ls in zip(['Any visit', 'Last visit'], ['--', '-']):
        ax.plot(np.NaN, np.NaN, ls=ls,
                label=ft_type, c='black', linewidth=3.5, markersize=11, marker='D', zorder=0)
    ax2 = ax.twinx()

    for group, color in zip(labels, colors):
        ax2.plot(np.NaN, np.NaN, c=color, label=group, linewidth=3.5, markersize=11, marker='D', zorder=5)

    ax2.get_yaxis().set_visible(False)
    ax2.get_xaxis().set_visible(False)

    ax2.legend(fontsize=35)
    ax.legend(fontsize=35, loc='upper center')
    plt.tight_layout()
    plt.savefig(figure_name)
    plt.clf()
