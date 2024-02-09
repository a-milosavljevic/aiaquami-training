"""
confusion_matrix.py script is by traing.py to generate confusion matrices.
You do not have to run this script.
"""
from model import *
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
import math


def plot_confusion_matrix(y_true, y_pred, normalize=False, title=None, cmap=plt.cm.Blues, show_zeros=False,
                          figsize=(10, 8.5)):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = 100 * cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Patch for matplotlib 3.1.1
    ax.set_ylim(cm.shape[0] - 0.5, -0.5)

    # Loop over data dimensions and create text annotations.
    fmt = '.1f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if show_zeros or cm[i, j] != 0:
                if normalize and cm[i, j] == 100:
                    ax.text(j, i, '100%',
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")
                else:
                    ax.text(j, i, format(cm[i, j], fmt) + ('%' if normalize else ''),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")
            else:
                ax.text(j, i, '-',
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.5)
    return ax


def plot_confusion_matrix_mean_std(cm_mean, cm_std, title=None, cmap=plt.cm.Blues, show_zeros=False, figsize=(10, 8.5)):
    if not title:
        title = 'Normalized confusion matrix'

    print(cm_mean)
    print(cm_std)

    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    im = ax.imshow(cm_mean, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm_mean.shape[1]),
           yticks=np.arange(cm_mean.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Patch for matplotlib 3.1.1
    ax.set_ylim(cm_mean.shape[0] - 0.5, -0.5)

    # Loop over data dimensions and create text annotations.
    thresh = cm_mean.max() / 2.
    for i in range(cm_mean.shape[0]):
        for j in range(cm_mean.shape[1]):
            if show_zeros or cm_mean[i, j] != 0:
                ax.text(j, i, format(cm_mean[i, j], '.1f') + '%\n±' + format(cm_std[i, j], '.1f'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
            else:
                ax.text(j, i, '-',
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.5)
    return ax


########################################################################################################################
# LOCAL EXECUTION
########################################################################################################################

if __name__ == "__main__":
    # Load model
    model_path = os.path.join(tmp_folder, 'model.h5')

    model = tf.keras.models.load_model(model_path)
    model.summary()

    # Prepare data
    if join_test_with_train:
        cm_images = val_images
        cm_classes = val_classes
    else:
        cm_images = test_images
        cm_classes = test_classes

    # Run prediction
    cm_gen = DataGenerator(images=cm_images, image_classes=cm_classes, use_augmentation=False)
    y_cm_raw = model.predict(cm_gen)
    y_cm_out = np.argmax(y_cm_raw, axis=1)
    y_cm = np.array(cm_classes, dtype=np.int32)
    misclassified = np.sum(y_cm != y_cm_out)
    accuracy = round(100 * (len(y_cm) - misclassified) / len(y_cm), 2)

    # Calc. data for confusion matrices
    cm = confusion_matrix(y_cm, y_cm_out)
    cm = 100 * cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plot confusion matrices
    np.set_printoptions(precision=1)

    cm = confusion_matrix(y_cm, y_cm_out)
    cm = 100 * cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig_size = round(len(classes) * 0.5)
    fig_size = (5 + fig_size, fig_size)

    plot_confusion_matrix(y_cm, y_cm_out, figsize=fig_size,
                          title='Misclassified {} out of {} specimens'.format(misclassified, len(y_cm)))
    plt.savefig(os.path.join(tmp_folder, 'cm.png'))

    plot_confusion_matrix(y_cm, y_cm_out, normalize=True, figsize=fig_size,
                          title='Classification accuracy {}%'.format(accuracy))
    plt.savefig(os.path.join(tmp_folder, 'cm_norm.png'))

    stop_threads()
