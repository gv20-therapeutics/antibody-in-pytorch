import io
import matplotlib.pyplot as plt
import numpy as np
import itertools
import scipy
import PIL.Image
from torchvision.transforms import ToTensor
from sklearn.metrics import roc_curve, roc_auc_score
import seaborn as sns

COLOR_PALETTE = 'bright'
sns.set_palette(COLOR_PALETTE)


def plot_to_image(figure):
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
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes

    source: https://www.tensorflow.org/tensorboard/image_summaries
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
    fpr, tpr, thresh = roc_curve(labels, scores)
    auroc = roc_auc_score(labels, scores)
    roc = sns.lineplot(x=fpr, y=tpr, label=f'{legend_label} -- {round(auroc, 3)}', ci=None)
    return roc


def plot_roc_curves(scores_list, labels_list, legend_labels_list, title='', save_path=None, dpi=300):
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
    plt.show()
    return roc


def roc_from_models(models, data_loaders,evaluate=True, title='', save_path=None, dpi=300):
    '''
    Evaluates set of models on data_loaders and plots overlaid ROC curves.

    Args:
        models (dict of str: torch.nn.Module):
            Dict mapping model names to Torch models. Each model must output 2 logits, the 2nd (index 1) of which
            will be used as a classification score for ROC plotting. Furthermore, each model must implement `predict`
            method which given a data_loader, returns tuple of (outputs, labels, loss). If `evaluate`, each model must
            also implement `evaluate` method.

        data_loaders (dict of str: torch.utils.data.DataLoader):
            Dict of data_loaders to evaluate models on. Must have same keys as `models`. of same length as `models`.
            `models[k]` is evaluated on `data_loaders[k]` for each key k.

    Returns (seaborn.lineplot): Plot with one ROC curve for each model in `models`, evaluated on `data_loader`.
    '''
    roc_scores = []
    roc_labels = []
    roc_legend_labels = []
    for model_name, model in models.items():
        outputs, labels, loss = model.predict(data_loaders[model_name])
        if evaluate:
            model.evaluate(outputs, labels)
        roc_scores.append(
            outputs[:, 1])  # extract output column containing 2nd logit, which represents probability of the 1-class
        roc_labels.append(labels)
        roc_legend_labels.append(model_name)
    roc = plot_roc_curves(roc_scores, roc_labels, roc_legend_labels, title=title, save_path=save_path, dpi=dpi)
    return roc
