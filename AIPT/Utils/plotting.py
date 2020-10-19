import io
import matplotlib.pyplot as plt
import numpy as np
import itertools
import scipy
import PIL.Image
from torchvision.transforms import ToTensor

def plot_to_image(figure):
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    plt.close(figure)
    return image

def plot_to_image2(figure):
    """
    Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    
    source: https://www.tensorflow.org/tensorboard/image_summaries
    """
    
    buf = io.BytesIO()
    
    # Use plt.savefig to save the plot to a PNG in memory.
    plt.savefig(buf, format='png')
    
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    im = scipy.misc.imread(buf)
    buf.close()
    return im

# def plot_to_image(figure):
#     """
#     Takes a matplotlib figure handle and converts it using
#     canvas and string-casts to a numpy array that can be
#     visualized in TensorBoard using the add_image function

#     Parameters:
#         writer (tensorboard.SummaryWriter): TensorBoard SummaryWriter instance.
#         fig (matplotlib.pyplot.fig): Matplotlib figure handle.
#         step (int): counter usually specifying steps/epochs/time.
    
#     source: http://martin-mundt.com/tensorboard-figures/
#     """

#     # Draw figure on canvas
#     figure.canvas.draw()

#     # Convert the figure to numpy array, read the pixel values and reshape the array
#     img = np.fromstring(figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
#     img = img.reshape(figure.canvas.get_width_height()[::-1] + (3,))

#     # Normalize into 0-1 range for TensorBoard(X). Swap axes for newer versions where API expects colors in first dim
#     img = img / 255.0
#     img = np.swapaxes(img, 0, 2) # if your TensorFlow + TensorBoard version are >= 1.8
# #     plt.close(figure)
#     return img


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

