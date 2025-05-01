from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd


def show_log_train(data_src, target_src, do_save=True, do_show=False):
    """
        plots the confusion matrix as and image
        :param data_src: location of data
        :param target_src : location of saved image
        :param do_save: saves the image
        :param do_show: shows the image
        :return: None (just plots)
        """
    data = pd.read_csv(data_src, delimiter=';')
    ls_train_acc = data['train_acc'].to_list()
    ls_train_loss = data['train_loss'].to_list()
    ls_test_acc = data['test_acc'].to_list()
    ls_test_class_acc = data['test_class_acc'].to_list()
    ls_test_loss = data['test_loss'].to_list()
    num_epoch = len(ls_train_acc)

    # Plot results
    fig, axs = plt.subplots(2, 1, sharex=True)

    # plot accuracies
    axs[0].plot(np.arange(num_epoch), ls_train_acc, label='train')
    axs[0].plot(np.arange(num_epoch), ls_test_acc, label='eval')
    axs[0].plot(np.arange(num_epoch), ls_test_class_acc, label='eval class')
    axs[0].set_title('Accuracy')
    axs[0].set_ylabel('Accuracy value [-]')
    axs[0].set_ylim(None, 1.0)
    axs[0].legend(loc='upper left')

    # plot losses
    axs[1].plot(np.arange(num_epoch), ls_train_loss, label='train')
    axs[1].plot(np.arange(num_epoch), ls_test_loss, label='eval')
    axs[1].set_title('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss value [-]')
    axs[1].legend()

    if do_save:
        plt.savefig(target_src + '/acc_loss_evolution.png')

    if do_show:
        plt.show()

    plt.clf()


def show_confusion_matrix(target_src, y_pred, y_true, class_labels, epoch=0, do_save=True, do_show=False):
    """
        plots the confusion matrix as and image
        :param target_src : location of saved image
        :param y_true: list of the GT label of the models
        :param y_pred: List of the predicted label of the models
        :param class_labels: List of strings containing the label tags
        :param epoch: number of the epoch of training which provided the results
        :param do_save: saves the image
        :param do_show: shows the image
        :return: None (just plots)
        """
    n_classes = len(class_labels)
    conf_mat = confusion_matrix(y_true, y_pred, labels=range(0, n_classes), normalize='true')
    df_conf_mat = pd.DataFrame(conf_mat, index=class_labels, columns=class_labels)

    fig = plt.figure()
    sn.heatmap(df_conf_mat, annot=True, cmap=sn.color_palette("Blues", as_cmap=True))
    ax = plt.gca()
    ax.set_title('Confusion Matrix - epoch ' + str(epoch + 1))

    plt.tight_layout()
    plt.ylabel('True labels')
    plt.xlabel('Predicted labels')
    fig.tight_layout()

    if do_save:
        plt.savefig(target_src + '/confustion_matrix.png')

    if do_show:
        plt.show()

    plt.clf()


def show_grid_search(target_src, data, title, do_save=True, do_show=False):
    """
        plots the results of grid search on kernel
        :param target_src : location of saved image
        :param data: values to plot and values of hyperparameters
        :param title: title of the figure
        :param do_save: saves the image
        :param do_show: shows the image
        :return: None (just plots)
        """
    fig, ax = plt.subplots()
    X = data.iloc[:, 0].unique()
    Y = data.iloc[:, 1].unique()
    Z = np.zeros((len(X), len(Y)))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            Z[j, i] = data[(data.iloc[:, 0] == x) & (data.iloc[:, 1] == y)].iloc[0, 2]
    cmap_reversed = cm.get_cmap('coolwarm').reversed()
    c = plt.pcolor(X, Y, Z, shading='auto', cmap=cmap_reversed)
    ax.set_title(title)
    ax.set_xlabel('Kernel size')
    ax.set_ylabel('number of repetition')
    ax.set_xticks(X)
    ax.set_yticks(Y)
    cbar = fig.colorbar(c, ax=ax)
    cbar.set_label('Accuracy [-]')
    if do_save:
        plt.savefig(f'{target_src}grid_search_{title.replace(" ", "_")}.png')

    if do_show:
        plt.show()

    plt.clf()


if __name__ == '__main__':
    print("Not the right way to use me Pal")
