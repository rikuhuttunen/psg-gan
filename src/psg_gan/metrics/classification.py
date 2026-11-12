import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pathlib import Path
from sklearn.metrics import cohen_kappa_score, classification_report, confusion_matrix


def plot_cm(cm, class_names, figsize = (10,7),
            fontsize=14, ax=None, show_yticks=True, show_xticks=True,
            xlabel='Predicted label', ylabel='True label'):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    
    Arguments
    ---------
    cm: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.
        
    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    # Calculate the sum of true labels for each class and broadcast to cm shape
    row_sums = np.tile(cm.sum(axis=1)[:, np.newaxis], (1, cm.shape[0]))
    # Set normalized to 0.0 if there are zero true samples
    norm_cm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype='float'), where=row_sums!=0)
    norm_cm_df = pd.DataFrame(norm_cm, index=class_names, columns=class_names)
    
    # Create annotations in the format of '<normalized>\n(<count>)'
    formatter = "{0:.2f}".format
    annot = np.array([formatter(f) + '\n(' + str(d) + ')' for f, d in
                      zip(norm_cm.flatten(), cm.flatten())]).reshape(cm.shape)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, layout='constrained')
    else:
        # Return None if the ax was given as an argument
        fig = None
    
    heatmap = sns.heatmap(norm_cm_df, annot=annot, ax=ax, cmap=plt.cm.Blues, cbar=False,
                              fmt='s', vmin=0.0, vmax=1.0, annot_kws={'size': fontsize})

    if show_yticks:
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0,
                                     ha='right', fontsize=fontsize)
    else:
        heatmap.yaxis.set_ticks([])

    if show_xticks:
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45,
                                     ha='right', fontsize=fontsize)
    else:
        heatmap.xaxis.set_ticks([])

    _ = ax.set_ylabel(ylabel, fontsize=fontsize+2)
    _ = ax.set_xlabel(xlabel, fontsize=fontsize+2)
    return fig


def plot_metrics(
        true: np.array,
        pred: np.array,
        class_names: list[str],
        save_dir: Path,
        save_prefix: str,
        cm_xlabel: str = 'Predicted',
        cm_ylabel: str = 'True',
        cm_figsize: tuple[float, float] = (9, 7)) -> None:
    cm = confusion_matrix(true, pred)
    fig = plot_cm(cm, class_names, xlabel=cm_xlabel, ylabel=cm_ylabel, figsize=cm_figsize)
    fig.savefig(save_dir / f'{save_prefix}_cm.pdf', dpi=300, bbox_inches='tight')

    clf_report = classification_report(true, pred, target_names=class_names, output_dict=True)
    cohen_kappa = cohen_kappa_score(true, pred)
    clf_report['cohen_kappa'] = cohen_kappa
    with open(save_dir / f'{save_prefix}_clf_report.json', 'w') as f:
        json.dump(clf_report, f, indent=2)
