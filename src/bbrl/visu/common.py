# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import os

import matplotlib.pyplot as plt


def final_show(save_figure, plot, directory, figure_name, x_label, y_label, title):
    """
    Finalize all plots, adding labels and putting the corresponding file in the
    specified directory
    :param save_figure: boolean stating whether the figure should be saved
    :param plot: whether the plot should be shown interactively
    :param figure_name: the name of the file where to save the figure
    :param x_label: label on the x axis
    :param y_label: label on the y axis
    :param title: title of the figure
    :param directory: the path where to save the picture
    :return: nothing
    """
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    if save_figure:
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = directory + figure_name
        plt.savefig(filename)

    if plot:
        plt.show()

    plt.close()
