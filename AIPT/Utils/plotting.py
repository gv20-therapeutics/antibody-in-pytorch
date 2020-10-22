import io
import matplotlib.pyplot as plt
import numpy as np
import itertools
import scipy
import PIL.Image
from torchvision.transforms import ToTensor
from sklearn.metrics import roc_curve
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
    roc = sns.lineplot(x=fpr, y=tpr, label=legend_label)
    return roc

def plot_roc_curves(scores_list, labels_list, legend_labels_list, title=''):
    assert len(scores_list) == len(labels_list)
    assert len(scores_list) == len(legend_labels_list)
    for scores, labels, legend_label in zip(scores_list, labels_list, legend_labels_list):
        roc = plot_roc_curve(scores, labels, legend_label=legend_label)
    roc.set(**{
        'title': title,
        'xlabel': 'FPR',
        'ylabel': 'TPR'
    })
    plt.show()
    return roc
