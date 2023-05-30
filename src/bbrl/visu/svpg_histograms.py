# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import matplotlib.pyplot as plt
import numpy as np

from bbrl.visu.common import final_show


def plot_histograms(
    rewards_list, labels, colors, title, directory, plot=True, save_figure=True
):
    n_bars = len(rewards_list)
    x = np.arange(len(rewards_list[0]))
    width = 0.5 / n_bars

    for i, rewards in enumerate(rewards_list):
        plt.bar(x + width * i, np.sort(rewards)[::-1], width=width, color=colors[i])

    plt.legend(labels=labels)
    plt.xticks([], [])

    figname = f"{title}-indep_vs_svpg.png"
    final_show(save_figure, plot, figname, "", "rewards", title, directory)
