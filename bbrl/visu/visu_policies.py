# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import random

import matplotlib.pyplot as plt
import numpy as np
import torch as th

from bbrl.visu.common import final_show


def plot_policy(
    agent, env, directory, env_name, best_reward, plot=False, stochastic=False
):
    if "cartpole" in env_name.lower():
        plot_env = plot_cartpole_policy
    elif "pendulum" in env_name.lower():
        plot_env = plot_pendulum_policy
    elif "lunarlander" in env_name.lower():
        plot_env = plot_lunarlander_policy
    else:
        plot_env = plot_standard_policy
    save_figure = True
    figure_name = f"policy_{env_name}_{best_reward}.png"
    plot_env(agent, env, directory, figure_name, plot, save_figure, stochastic)


def plot_pendulum_policy(
    agent, env, directory, figure_name, plot=True, save_figure=True, stochastic=None
):
    """
    Plot an agent for the Pendulum environment
    :param agent: the policy specifying the action to be plotted
    :param env: the evaluation environment
    :param figure_name: the name of the file to save the figure
    :param directory: the path to the file to save the figure
    :param plot: whether the plot should be interactive
    :param save_figure: whether the figure should be saved
    :param stochastic: whether one wants to plot a deterministic or stochastic policy
    :return: nothing
    """
    if env.observation_space.shape[0] <= 2:
        msg = f"Observation space dim {env.observation_space.shape[0]}, should be > 2"
        raise (ValueError(msg))
    definition = 100
    portrait = np.zeros((definition, definition))
    state_min = env.observation_space.low
    state_max = env.observation_space.high

    for index_t, t in enumerate(np.linspace(-np.pi, np.pi, num=definition)):
        for index_td, td in enumerate(
            np.linspace(state_min[2], state_max[2], num=definition)
        ):
            obs = np.array([[np.cos(t), np.sin(t), td]])
            obs = th.from_numpy(obs.astype(np.float32))
            action = agent.predict_action(obs, stochastic)

            portrait[definition - (1 + index_td), index_t] = action.item()

    plt.figure(figsize=(10, 10))
    plt.imshow(
        portrait,
        cmap="inferno",
        extent=[-np.pi, np.pi, state_min[2], state_max[2]],
        aspect="auto",
    )

    title = "Pendulum Actor"
    plt.colorbar(label="action")
    directory += "/pendulum_policies/"
    # Add a point at the center
    plt.scatter([0], [0])
    x_label, y_label = getattr(env.observation_space, "names", ["x", "y"])
    final_show(save_figure, plot, directory, figure_name, x_label, y_label, title)


def plot_cartpole_policy(
    agent, env, directory, figure_name, plot=True, save_figure=True, stochastic=None
):
    """
    Visualization of a policy in a N-dimensional state space
    The N-dimensional state space is projected into its first two dimensions.
    A FeatureInverter wrapper should be used to select which features to put first to plot them
    :param agent: the policy agent to be plotted
    :param env: the environment
    :param figure_name: the name of the file to save the figure
    :param directory: the path to the file to save the figure
    :param plot: whether the plot should be interactive
    :param save_figure: whether the figure should be saved
    :param stochastic: whether one wants to plot a deterministic or stochastic policy
    :return: nothing
    """
    if env.observation_space.shape[0] <= 2:
        msg = f"Observation space dim {env.observation_space.shape[0]}, should be > 2"
        raise (ValueError(msg))
    definition = 100
    portrait = np.zeros((definition, definition))
    state_min = env.observation_space.low
    state_max = env.observation_space.high

    for index_x, x in enumerate(
        np.linspace(state_min[0], state_max[0], num=definition)
    ):
        for index_y, y in enumerate(
            np.linspace(state_min[2], state_max[2], num=definition)
        ):
            obs = np.array([x])
            z1 = random.random() - 0.5
            z2 = random.random() - 0.5
            obs = np.append(obs, z1)
            obs = np.append(obs, y)
            obs = np.append(obs, z2)
            obs = th.from_numpy(obs.astype(np.float32))
            action = agent.predict_action(obs, stochastic)

            portrait[definition - (1 + index_y), index_x] = action.item()

    plt.figure(figsize=(10, 10))
    plt.imshow(
        portrait,
        cmap="inferno",
        extent=[state_min[0], state_max[0], state_min[2], state_max[2]],
        aspect="auto",
    )

    title = "Cartpole Actor"
    plt.colorbar(label="action")
    directory += "/cartpole_policies/"
    # Add a point at the center
    plt.scatter([0], [0])
    x_label, y_label = getattr(env.observation_space, "names", ["x", "y"])
    final_show(save_figure, plot, directory, figure_name, x_label, y_label, title)


def plot_lunarlander_policy(
    agent, env, directory, figure_name, plot=True, save_figure=True, stochastic=None
):
    """
    Visualization of a policy in a N-dimensional state space
    The N-dimensional state space is projected into its first two dimensions.
    A FeatureInverter wrapper should be used to select which features to put first to plot them
    :param agent: the policy agent to be plotted
    :param env: the environment
    :param figure_name: the name of the file to save the figure
    :param directory: the path to the file to save the figure
    :param plot: whether the plot should be interactive
    :param save_figure: whether the figure should be saved
    :param stochastic: whether one wants to plot a deterministic or stochastic policy
    :return: nothing
    """
    if env.observation_space.shape[0] <= 2:
        msg = f"Observation space dim {env.observation_space.shape[0]}, should be > 2"
        raise (ValueError(msg))
    definition = 100
    portrait = np.zeros((definition, definition))
    state_min = [-1.5, -1.5, -5.0, -5.0, -3.14, -5.0, -0.0, -0.0]
    state_max = [1.5, 1.5, 5.0, 5.0, 3.14, 5.0, 1.0, 1.0]

    for index_x, x in enumerate(
        np.linspace(state_min[0], state_max[0], num=definition)
    ):
        for index_y, y in enumerate(
            np.linspace(state_min[2], state_max[1], num=definition)
        ):
            obs = np.array([x])
            obs = np.append(obs, y)
            nb = range(len(state_min) - 2)
            for _ in nb:
                z = random.random() - 0.5
                obs = np.append(obs, z)
            obs = th.from_numpy(obs.astype(np.float32))
            action = agent.predict_action(obs, stochastic)

            portrait[definition - (1 + index_y), index_x] = action.item()

    plt.figure(figsize=(10, 10))
    plt.imshow(
        portrait,
        cmap="inferno",
        extent=[state_min[0], state_max[0], state_min[2], state_max[2]],
        aspect="auto",
    )

    title = "Actor"
    plt.colorbar(label="action")
    directory += "/policies/"
    # Add a point at the center
    plt.scatter([0], [0])
    x_label, y_label = getattr(env.observation_space, "names", ["x", "y"])
    final_show(save_figure, plot, directory, figure_name, x_label, y_label, title)


def plot_standard_policy(
    agent, env, directory, figure_name, plot=True, save_figure=True, stochastic=None
):
    """
    Visualization of a policy in a N-dimensional state space
    The N-dimensional state space is projected into its first two dimensions.
    A FeatureInverter wrapper should be used to select which features to put first to plot them
    :param agent: the policy agent to be plotted
    :param env: the environment
    :param figure_name: the name of the file to save the figure
    :param directory: the path to the file to save the figure
    :param plot: whether the plot should be interactive
    :param save_figure: whether the figure should be saved
    :param stochastic: whether one wants to plot a deterministic or stochastic policy
    :return: nothing
    """
    if env.observation_space.shape[0] <= 2:
        msg = f"Observation space dim {env.observation_space.shape[0]}, should be > 2"
        raise (ValueError(msg))
    definition = 100
    portrait = np.zeros((definition, definition))
    state_min = env.observation_space.low
    state_max = env.observation_space.high
    for i in range(len(state_min)):
        if state_min[i] == -np.inf:
            state_min[i] = -1e20
        if state_max[i] == np.inf:
            state_max[i] = 1e20

    for index_x, x in enumerate(
        np.linspace(state_min[0], state_max[0], num=definition)
    ):
        for index_y, y in enumerate(
            np.linspace(state_min[2], state_max[1], num=definition)
        ):
            obs = np.array([x])
            obs = np.append(obs, y)
            nb = range(len(state_min) - 2)
            for _ in nb:
                z = random.random() - 0.5
                obs = np.append(obs, z)
            obs = th.from_numpy(obs.astype(np.float32))
            action = agent.predict_action(obs, stochastic)

            portrait[definition - (1 + index_y), index_x] = action.item()

    plt.figure(figsize=(10, 10))
    plt.imshow(
        portrait,
        cmap="inferno",
        extent=[state_min[0], state_max[0], state_min[2], state_max[2]],
        aspect="auto",
    )

    title = "Actor"
    plt.colorbar(label="action")
    directory += "/policies/"
    # Add a point at the center
    plt.scatter([0], [0])
    x_label, y_label = getattr(env.observation_space, "names", ["x", "y"])
    final_show(save_figure, plot, directory, figure_name, x_label, y_label, title)
