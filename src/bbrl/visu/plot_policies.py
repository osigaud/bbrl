# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import random
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import random
import torch
from gymnasium.spaces import flatdim

from bbrl import Agent
from bbrl.agents.gymnasium import GymAgent
from bbrl.visu.common import final_show
from bbrl.workspace import Workspace


def plot_policy(
    actor: Agent,
    env: GymAgent,
    best_reward,
    directory: str,
    env_name: Union[str, None] = None,
    plot: bool = False,
    save_fig: bool = True,
    definition: int = 100,
    var_name_obs: str = "env/env_obs",
    var_name_action: str = "action",
    **kwargs,
) -> None:
    """
    Plot the policy of the agent
    :param Agent actor: the agent
    :param GymAgent env: the environment
    :param Tensor best_reward: the best reward
    :param str directory: the directory to save the figure
    :param str env_name: the name of the environment
    :param bool plot: if True, plot the figure
    :param bool save_fig: if True, save the figure
    :param int definition: the definition of the plot
    :param str var_name_obs: the name of the observation variable red by the agent
    :param str var_name_action: the name of the action variable written by the agent
    :param kwargs: the arguments to be passed to the actor forward function
    :return: None
    """
    if env_name is None:
        env_name = env.envs[0].unwrapped.spec.id

    assert (
        len(env.observation_space.shape) == 1
    ), "Nested observation space not supported"

    if env.observation_space.shape[0] < 2:
        msg = f"Observation space dim {env.observation_space.shape[0]}, should be >= 2"
        raise (ValueError(msg))

    state_min = env.observation_space.low
    state_max = env.observation_space.high

    for i in range(len(state_min)):
        if state_min[i] == -np.inf:
            state_min[i] = -1e20
        if state_max[i] == np.inf:
            state_max[i] = 1e20

    workspace = Workspace()

    if env.is_continuous_action():
        action_dim = flatdim(env.action_space)
    else:
        action_dim = 1

    all_obs = []
    for index_x, x in enumerate(np.linspace(state_min[0], state_max[0], definition)):
        for index_y, y in enumerate(
            np.linspace(state_min[1], state_max[1], definition)
        ):
            # create possible states to observe
            obs = [x, y]
            for i in range(2, env.observation_space.shape[0]):
                # generate randomness around mean
                z = random.random() - 0.5
                # z = random.uniform(state_min[i], state_max[i])
                obs.append(z)
            all_obs.append(obs)
    all_obs = torch.tensor([all_obs], dtype=torch.float32)

    workspace.set_full(var_name_obs, all_obs, batch_dims=None)

    # predictions ici de l'action selon la policy
    actor(workspace, t=0, **kwargs)

    # get the actions from the workspace
    portrait = (
        workspace.get_full(var_name_action)
        .reshape(definition, definition, action_dim)
        .detach()
        .numpy()
    )

    for dim in range(action_dim):
        portrait = portrait[:, :, dim]

        plt.figure(figsize=(10, 10))
        plt.imshow(
            portrait,
            cmap="inferno",
            extent=[state_min[0], state_max[0], state_min[1], state_max[1]],
            aspect="auto",
        )

        figure_name: str = f"policy_{env_name}_dim_{dim}_{best_reward}.png"

        title = "{} Actor (action dim: {})".format(env_name, dim)
        plt.colorbar(label="action")
        directory += "/" + env_name + "_policies/"
        # Add a point at the center
        plt.scatter([0], [0])
        x_label, y_label = getattr(env.observation_space, "names", ["x", "y"])
        final_show(save_fig, plot, directory, figure_name, x_label, y_label, title)
