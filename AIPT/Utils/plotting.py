import io
import itertools
import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score
from torchvision.transforms import ToTensor
from AIPT.Utils.metrics import binary_classification_metrics

COLOR_PALETTE = 'bright'
sns.set_palette(COLOR_PALETTE)


def plot_to_image(figure):
    '''
    Converts matplotlib figure to PyTorch image tensor suitable for logging in TensorBoard.

    Args:
        figure (matplotlib.pyplot.figure): Matplotlib figure to convert to tensor.

    Returns (torch.Tensor): Tensor containing image pixels.
    '''
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    plt.close(figure)
    return image


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
        cm (array, shape=[n, n]): a confusion matrix of integer classes
        class_names (array, shape=[n]): String names of the integer classes

    Source: https://www.tensorflow.org/tensorboard/image_summaries

    Returns (matplotlib.pyplot.figure): Heatmap-style confusion matrix.

    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Compute the labels from the normalized confusion matrix.
    labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
    return figure


def plot_roc_curve(scores, labels, legend_label=None):
    '''
    Plots ROC curve (overlays onto current plot).

    Args:
        scores (list of float): List of scores.
        labels (list of int in {0,1}): List of class labels, parallel to `scores`.
        legend_label: Name of series defined by `scores` and `labels`.

    Returns: ROC defined by `scores` and `labels`, with legend labeled with `legend_label` and AUROC.

    '''
    fpr, tpr, thresh = roc_curve(labels, scores)
    auroc = roc_auc_score(labels, scores)
    roc = sns.lineplot(x=fpr, y=tpr, label=f'{legend_label} -- {round(auroc, 3)}', ci=None)
    return roc


def plot_roc_curves(scores_list, labels_list, legend_labels_list, title='', save_path=None, dpi=300, show=True):
    '''
    Plots overlaid ROC curves defined by parallel lists of scores and labels.

    Args:
        scores_list (list of list of float):
            Each inner list in `scores_list` is a list of scores that together with a corresponding label list
            defines an ROC curve.

        labels_list (list of list of int in {0,1}):
            `labels_list[i]` contains class labels corresponding to `scores_list[i]`.

        legend_labels_list (list of str):
            `legend_labels_list[i]` contains the name of the series defined by `scores_list[i]` and `labels_list[i]`.

        title (str): Plot title.
        save_path (str): Plot save path.
        dpi (int): Plot DPI.
        show (bool): If True, display the plot.
    '''
    assert len(scores_list) == len(labels_list)
    assert len(scores_list) == len(legend_labels_list)

    for scores, labels, legend_label in zip(scores_list, labels_list, legend_labels_list):
        roc = plot_roc_curve(scores, labels, legend_label=legend_label)
    roc.set(**{
        'title': title,
        'xlabel': 'FPR',
        'ylabel': 'TPR'
    })
    if save_path is not None:
        plt.savefig(save_path, dpi=dpi)
        print(f'Saved to: {save_path}')
    if show:
        plt.show()
    plt.clf()


def roc_from_models(models, data_loaders, title='', save_path=None, dpi=300, show=True, print_metrics=True):
    '''
    Evaluates set of models on data_loaders and plots overlaid ROC curves.

    Args:
        models (dict of str: torch.nn.Module):
            Dict mapping model names to Torch models. Each model must output 2 logits, the 2nd (index 1) of which
            will be used as a classification score for ROC plotting. Furthermore, each model must implement `predict`
            method which given a data_loader, returns tuple of (outputs, labels, loss).

        data_loaders (dict of str: torch.utils.data.DataLoader):
            Dict of data_loaders to evaluate models on. Must have same keys as `models`. of same length as `models`.
            `models[k]` is evaluated on `data_loaders[k]` for each key k.

        title, save_path, dpi, show: arguments passed to `plot_roc_curves`

        print_metrics (bool): If True, compute and print loss, confusion matrix, accuracy, and MCC.
    '''
    roc_scores = []
    roc_labels = []
    roc_legend_labels = []
    for model_name, model in models.items():
        if print_metrics:
            print(f'{model_name}:')
        metrics, outputs, labels = model.evaluate(data_loaders[model_name], verbose=print_metrics)
        roc_scores.append(
            outputs[:, 1])  # extract output column containing 2nd logit, which represents probability of the 1-class
        roc_labels.append(labels)
        roc_legend_labels.append(model_name)
    roc = plot_roc_curves(roc_scores, roc_labels, roc_legend_labels, title=title, save_path=save_path, dpi=dpi, show=show)
