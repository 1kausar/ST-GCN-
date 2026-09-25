import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_cm(CM, normalize=True, save_dir='', names_x=(), names_y=(), show=False):
    """
    Plot + save confusion matrix.
    :param CM: array (rows = predicted, cols = true)   [sklearn cm ke .T kore dite hobe]
    :param normalize: column-wise 0-1 normalize
    :param save_dir: folder
    :param names_x / names_y: class names
    """
    import seaborn as sn
    CM = np.asarray(CM)
    array = CM / ((CM.sum(0).reshape(1, -1) + 1E-6) if normalize else 1)
    fmt = 'd'
    if normalize:
        fmt = '.2f'
        array[array < 0.005] = np.nan
    else:
        array = np.asarray(array, dtype='int')

    n = CM.shape[0]
    fig = plt.figure(figsize=(max(6, 2.2 * n + 2), max(5, 2.0 * n + 1.5)), tight_layout=True)
    sn.set_theme(font_scale=1.4)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ax = sn.heatmap(array, annot=True, annot_kws={'size': 16}, cmap='Blues', fmt=fmt,
                        square=True,
                        xticklabels=list(names_x) if len(names_x) == n else 'auto',
                        yticklabels=list(names_y) if len(names_y) == n else 'auto')
        ax.set_facecolor((1, 1, 1))
    acc = np.trace(CM) / max(CM.sum(), 1)
    ax.set_xlabel('True', fontweight='bold', fontsize=16)
    ax.set_ylabel('Predicted', fontweight='bold', fontsize=16)
    ax.set_title('Accuracy: {:.1f}%'.format(acc * 100), fontweight='bold', fontsize=16)
    if show:
        plt.show()
    name = 'confusion_matrix_normalize.png' if normalize else 'confusion_matrix.png'
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(save_dir) / name, dpi=200)
    plt.close(fig)
