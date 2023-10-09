# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import warnings
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import random
import torch
from gymnasium import Space
from gymnasium.core import ActType

from bbrl import Agent
from bbrl.agents.gymnasium import GymAgent
from bbrl.visu.common import final_show
from bbrl.workspace import Workspace

# The plot critic actions below could probably be factored or at least reuse common subparts


# plot a DPDG-like critic.
# If the input_action is None, which cannot be the case with DDPG-like critic, a random action is drawn, which makes little sense.

def plot_critic(
    agent: Agent,
    env: GymAgent,
    best_reward,
    directory: str,
    env_name: Union[str, None] = None,
    plot: bool = False,
    save_fig: bool = True,
    definition: int = 100,
    input_action=None,
    var_name_obs: str = "env/env_obs",
    var_name_action: str = "action",
    var_name_value: str = "None",
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

    if not agent.is_q_function and input_action is not None:
        warnings.warn("action is ignored for non q function agent")
    if agent.is_q_function and input_action is None:
        action_space: Space[ActType] = env.get_action_space()
        input_action = action_space.sample()   
        warnings.warn("trying to plot a Q critic without giving an action")
        
    assert (
        len(env.observation_space.shape) == 1
    ), "Nested observation space not supported"

    if env.observation_space.shape[0] < 2:
        msg = f"Observation space dim {env.observation_space.shape[0]}, should be >= 2"
        raise (ValueError(msg))

    state_min = env.observation_space.low
    state_max = env.observation_space.high

    # TODO: it would be better to determine the min and max from the available data...
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
                # generate randomness around mean
                z = random.random() - 0.5
                # z = np.random.uniform(state_min[i], state_max[i])
                obs = np.append(obs, z)
            all_obs.append(obs)
    all_obs = torch.tensor([all_obs], dtype=torch.float32)

    workspace.set_full(var_name_obs, all_obs, batch_dims=None)

    if agent.is_q_function:
        action = torch.tensor([[input_action for _ in range(definition**2)]])
        workspace.set_full(var_name_action, action, batch_dims=None)
        if var_name_value == "None":
            var_name_value: str = f"{agent.name}/q_values"
    else:
        if var_name_value == "None":
            var_name_value: str = f"{agent.name}/v_values"

    agent(workspace, t=0, **kwargs)
    data = workspace.get_full(var_name_value)
    portrait_data = (
        data
        .reshape(definition, definition, env.action_space.n)
        .detach()
        .numpy()
    )
    portrait = portrait_data[input_action]
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

# Plot a DQN-like critic.
# If input_action is "policy", it plots the actions with a different color for each action
# If input_action is "None", it plots the value of the best action
# Otherwise, input_action is a number and it plots the Q-value of the corresponding action
    
def plot_discrete_q(
    agent: Agent,
    env: GymAgent,
    best_reward,
    directory: str,
    env_name: Union[str, None] = None,
    plot: bool = False,
    save_fig: bool = True,
    definition: int = 100,
    input_action=None,
    var_name_obs: str = "env/env_obs",
    var_name_action: str = "action",
    var_name_value: str = "None",
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
    :param ActType input_action: the action to use if the agent is a q function. Use "policy" if you want to see the critic as a policy (display the chosen action)
    :param str var_name_obs: the name of the observation variable
    :param str var_name_action: the name of the action variable
    :param kwargs: the arguments to be passed to the agent forward function
    :return: None
    """

    if env_name is None:
        env_name = env.envs[0].unwrapped.spec.id

    figure_name: str = f"critic_{env_name}_{best_reward}.png"

    if not agent.is_q_function and input_action is not None:
        warnings.warn("action is ignored for non q function agent")

    assert (
        len(env.observation_space.shape) == 1
    ), "Nested observation space not supported"

    if env.observation_space.shape[0] < 2:
        msg = f"Observation space dim {env.observation_space.shape[0]}, should be >= 2"
        raise (ValueError(msg))

    state_min = env.observation_space.low
    state_max = env.observation_space.high

    # TODO: it would be better to determine the min and max from the available data...
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
                # generate randomness around mean
                z = random.random() - 0.5
                # z = np.random.uniform(state_min[i], state_max[i])
                obs = np.append(obs, z)
            all_obs.append(obs)
    all_obs = torch.tensor([all_obs], dtype=torch.float32)

    workspace.set_full(var_name_obs, all_obs, batch_dims=None)

    assert agent.is_q_function, "plot_discrete_q should only be called for Q functions"
    
    if var_name_value == "None":
        var_name_value: str = f"{agent.name}/q_values"

    agent(workspace, t=0, **kwargs)
    data = workspace.get_full(var_name_value)

    if input_action is None:
        q_values = data.max(dim=-1).values
    elif input_action == "policy":
        q_values = data.max(dim=-1).indices
    else:
        q_values = data[:, :, input_action]
        
    portrait = (
        q_values.reshape(definition, definition)
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
