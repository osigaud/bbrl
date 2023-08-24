# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import warnings
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from gymnasium import Space
from gymnasium.core import ActType

from bbrl import Agent
from bbrl.agents.gymnasium import GymAgent
from bbrl.visu.common import final_show
from bbrl.workspace import Workspace


def plot_critic(
    agent: Agent,
    env: GymAgent,
    best_reward,
    directory: str,
    env_name: Union[str, None] = None,
    plot: bool = False,
    save_fig: bool = True,
    definition: int = 100,
    action=None,
    var_name_obs: str = "env/env_obs",
    var_name_action: str = "action",
    **kwargs,
) -> None:
    """
    Plot the critic of an agent
    :param Agent agent: the agent
    :param GymAgent env: the environment
    :param Tensor best_reward: the best reward
    :param str directory: the directory to save the figure
    :param str env_name: the name of the environment
    :param bool plot: if True, plot the figure
    :param bool save_fig: if True, save the figure
    :param int definition: the definition of the plot
    :param ActType action: the action to use if the agent is a q function
    :param str var_name_obs: the name of the observation variable
    :param str var_name_action: the name of the action variable
    :param kwargs: the arguments to be passed to the agent forward function
    :return: None
    """

    if env_name is None:
        env_name = env.envs[0].unwrapped.spec.id

    figure_name: str = f"critic_{env_name}_{best_reward}.png"

    if not agent.is_q_function and action is not None:
        warnings.warn("action is ignored for non q function agent")
    if agent.is_q_function and action is None:
        warnings.warn("action is None for q function agent, using random action")

    if agent.is_q_function:
        action_space: Space[ActType] = env.get_action_space()
        if action is None:
            action = action_space.sample()

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

    workspace: Workspace = Workspace()

    all_obs = []
    for index_x, x in enumerate(np.linspace(state_min[0], state_max[0], definition)):
        for index_y, y in enumerate(
            np.linspace(state_min[1], state_max[1], definition)
        ):
            # create possible states to observe
            obs = [x, y]
            for i in range(2, env.observation_space.shape[0]):
                z = np.random.uniform(state_min[i], state_max[i])
                obs = np.append(obs, z)
            all_obs.append(obs)
    all_obs = torch.tensor([all_obs], dtype=torch.float32)

    workspace.set_full(var_name_obs, all_obs, batch_dims=None)

    if agent.is_q_function:
        action = torch.tensor([[action for _ in range(definition**2)]])
        workspace.set_full(var_name_action, action, batch_dims=None)
        var_name_q_val: str = f"{agent.name}/q_values"
    else:
        var_name_q_val: str = f"{agent.name}/v_values"

    agent(workspace, t=0, **kwargs)
    print(workspace.variables)
    portrait = (
        workspace.get_full(var_name_q_val)
        .reshape(definition, definition)
        .detach()
        .numpy()
    )

    plt.figure(figsize=(10, 10))
    plt.imshow(
        portrait,
        cmap="inferno",
        extent=[
            state_min[0],
            state_max[0],
            state_min[1],
            state_max[1],
        ],
        aspect="auto",
    )

    directory += "/" + env_name + "_critics/"
    title = env_name + " critic"
    plt.colorbar(label="critic value")

    # Add a point at the center
    plt.scatter([0], [0])
    x_label, y_label = getattr(env.observation_space, "names", ["x", "y"])
    final_show(save_fig, plot, directory, figure_name, x_label, y_label, title)
