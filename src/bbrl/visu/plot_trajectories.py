# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np
import matplotlib.pyplot as plt
from bbrl.visu.common import final_show
from bbrl.workspace import Workspace

from typing import Union


def plot_trajectory(
        data,
        fig_index = 1,
        x_plot: Union[int, None] = 0,
        y_plot: Union[int, None] = 1,
        x_label: str = 'x',
        y_label: str = 'y',
        obs_key: str = 'env/obs',
        title: str = 'Trajectory',
        save_figure=True,
        plot=True
) -> None:

    if isinstance(data, Workspace):
        o = data.get_full(obs_key).numpy()
    elif isinstance(data, np.ndarray):
        o = data
    else:
        raise ValueError("data should be a Workspace or a numpy array")
    o = np.swapaxes(o, 1, 0)
    x = o[:, :, x_plot]
    y = o[:, :, y_plot]

    colors = []
    for c in range(o.shape[0]):
        colors += [c] * o.shape[1]

    plt.scatter(x, y, c=colors, cmap='viridis', alpha=0.7, s=1)
    figname = "trajectory_" + str(fig_index) + ".pdf"
    final_show(save_figure, plot, figname, x_label, y_label, title, "/plots/")
