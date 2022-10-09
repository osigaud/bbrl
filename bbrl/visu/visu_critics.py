# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import random

import matplotlib.pyplot as plt
import numpy as np
import torch as th

from bbrl.visu.common import final_show


def plot_critic(agent, env, directory, env_name, best_reward, plot=False) -> None:
    figure_name = f"critic_{env_name}_{best_reward}.png"
    if agent.is_q_function:
        if "cartpole" in env_name.lower():
            env_string = "CartPole"
            plot_env = plot_cartpole_critic_q
        elif "pendulum" in env_name.lower():
            env_string = "Pendulum"
            plot_env = plot_pendulum_critic_q
        elif "lunarlander" in env_name.lower():
            env_string = "LunarLander"
            plot_env = plot_lunarlander_critic_q
        else:
            env_string = env_name
            plot_env = plot_standard_critic_q
        plot_env(agent, env, env_string, directory, figure_name, plot, action=None)
    else:
        if "cartpole" in env_name.lower():
            env_string = "CartPole"
            plot_env = plot_cartpole_critic_v
        elif "pendulum" in env_name.lower():
            env_string = "Pendulum"
            plot_env = plot_pendulum_critic_v
        elif "lunarlander" in env_name.lower():
            env_string = "LunarLander"
            plot_env = plot_lunarlander_critic_v
        else:
            env_string = env_name
            plot_env = plot_standard_critic_v
        plot_env(agent, env, env_string, directory, figure_name, plot)


def plot_pendulum_critic_v(
    agent,
    env,
    env_string,
    directory,
    figure_name,
    plot=True,
    save_figure=True,
    stochastic=None,
):
    """
    Plot a critic for the Pendulum environment
    :param agent: the critic agent to be plotted
    :param env: the evaluation environment
    :param env_string: the name of the environment
    :param plot: whether the plot should be interactive
    :param directory: the directory where to save the figure
    :param figure_name: the name of the file to save the figure
    :param save_figure: whether the figure should be saved
    :param stochastic: whether we plot the deterministic or stochastic version
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

            value = agent.model(obs).squeeze(-1)

            portrait[definition - (1 + index_td), index_t] = value.item()

    plt.figure(figsize=(10, 10))
    plt.imshow(
        portrait,
        cmap="inferno",
        extent=[-180, 180, state_min[2], state_max[2]],
        aspect="auto",
    )

    directory += "/" + env_string + "_critics/"
    title = env_string + " Critic"
    plt.colorbar(label="critic value")

    # Add a point at the center
    plt.scatter([0], [0])
    x_label, y_label = getattr(env.observation_space, "names", ["x", "y"])
    final_show(save_figure, plot, directory, figure_name, x_label, y_label, title)


def plot_cartpole_critic_v(
    agent,
    env,
    env_string,
    directory,
    figure_name,
    plot=True,
    save_figure=True,
    stochastic=None,
):
    """
    Visualization of the critic in a N-dimensional state space
    The N-dimensional state space is projected into its first two dimensions.
    A FeatureInverter wrapper should be used to select which features to put first to plot them
    :param agent: the critic agent to be plotted
    :param env: the environment
    :param env_string: the name of the environment
    :param plot: whether the plot should be interactive
    :param directory: the directory where to save the figure
    :param figure_name: the name of the file where to plot the function
    :param save_figure: whether the plot should be saved into a file
    :param stochastic: whether we plot the deterministic or stochastic version
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
            obs = obs.reshape(1, -1)
            obs = th.from_numpy(obs.astype(np.float32))
            value = agent.model(obs).squeeze(-1)
            portrait[definition - (1 + index_y), index_x] = value.item()

    plt.figure(figsize=(10, 10))
    plt.imshow(
        portrait,
        cmap="inferno",
        extent=[state_min[0], state_max[0], state_min[2], state_max[2]],
        aspect="auto",
    )

    directory += "/" + env_string + "_critics/"
    title = env_string + " Critic"
    plt.colorbar(label="critic value")

    # Add a point at the center
    plt.scatter([0], [0])
    x_label, y_label = getattr(env.observation_space, "names", ["x", "y"])
    final_show(save_figure, plot, directory, figure_name, x_label, y_label, title)


def plot_lunarlander_critic_v(
    agent,
    env,
    env_string,
    directory,
    figure_name,
    plot=True,
    save_figure=True,
):
    """
    Visualization of the critic in a N-dimensional state space
    The N-dimensional state space is projected into its first two dimensions.
    A FeatureInverter wrapper should be used to select which features to put first to plot them
    :param agent: the critic agent to be plotted
    :param env: the environment
    :param env_string: the name of the environment
    :param plot: whether the plot should be interactive
    :param directory: the directory where to save the figure
    :param figure_name: the name of the file where to plot the function
    :param save_figure: whether the plot should be saved into a file
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
            np.linspace(state_min[1], state_max[1], num=definition)
        ):
            obs = np.array([x])
            obs = np.append(obs, y)
            nb = range(len(state_min) - 2)
            for _ in nb:
                z = random.random() - 0.5
                obs = np.append(obs, z)
            obs = obs.reshape(1, -1)
            obs = th.from_numpy(obs.astype(np.float32))
            value = agent.model(obs).squeeze(-1)
            portrait[definition - (1 + index_y), index_x] = value.item()

    plt.figure(figsize=(10, 10))
    plt.imshow(
        portrait,
        cmap="inferno",
        extent=[state_min[0], state_max[0], state_min[1], state_max[1]],
        aspect="auto",
    )

    directory += "/" + env_string + "_critics/"
    title = env_string + " Critic"
    plt.colorbar(label="critic value")

    # Add a point at the center
    plt.scatter([0], [0])
    x_label, y_label = getattr(env.observation_space, "names", ["x", "y"])
    final_show(save_figure, plot, directory, figure_name, x_label, y_label, title)


def plot_standard_critic_v(
    agent,
    env,
    env_string,
    directory,
    figure_name,
    plot=True,
    save_figure=True,
):
    """
    Visualization of the critic in a N-dimensional state space
    The N-dimensional state space is projected into its first two dimensions.
    A FeatureInverter wrapper should be used to select which features to put first to plot them
    :param agent: the critic agent to be plotted
    :param env: the environment
    :param env_string: the name of the environment
    :param plot: whether the plot should be interactive
    :param directory: the directory where to save the figure
    :param figure_name: the name of the file where to plot the function
    :param save_figure: whether the plot should be saved into a file
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
            np.linspace(state_min[1], state_max[1], num=definition)
        ):
            obs = np.array([x])
            obs = np.append(obs, y)
            nb = range(len(state_min) - 2)
            for _ in nb:
                z = random.random() - 0.5
                obs = np.append(obs, z)
            obs = obs.reshape(1, -1)
            obs = th.from_numpy(obs.astype(np.float32))
            value = agent.model(obs).squeeze(-1)
            portrait[definition - (1 + index_y), index_x] = value.item()

    plt.figure(figsize=(10, 10))
    plt.imshow(
        portrait,
        cmap="inferno",
        extent=[state_min[0], state_max[0], state_min[1], state_max[1]],
        aspect="auto",
    )

    directory += "/" + env_string + "_critics/"
    title = env_string + " Critic"
    plt.colorbar(label="critic value")

    # Add a point at the center
    plt.scatter([0], [0])
    x_label, y_label = getattr(env.observation_space, "names", ["x", "y"])
    final_show(save_figure, plot, directory, figure_name, x_label, y_label, title)


def plot_pendulum_critic_q(
    agent,
    env,
    env_string,
    directory,
    figure_name,
    plot=True,
    save_figure=True,
    action=None,
):
    """
    Plot a critic for the Pendulum environment
    :param agent: the critic agent to be plotted
    :param env: the evaluation environment
    :param env_string: the name of the environment
    :param plot: whether the plot should be interactive
    :param directory: the directory where to save the figure
    :param figure_name: the name of the file to save the figure
    :param save_figure: whether the figure should be saved
    :param action: the action for which we want to plot the value
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

            if action is None:
                action = th.Tensor([0])
            value = agent.predict_value(obs[0], action)

            portrait[definition - (1 + index_td), index_t] = value.item()

    plt.figure(figsize=(10, 10))
    plt.imshow(
        portrait,
        cmap="inferno",
        extent=[-180, 180, state_min[2], state_max[2]],
        aspect="auto",
    )

    directory += "/" + env_string + "_critics/"
    title = env_string + " Critic"
    plt.colorbar(label="critic value")

    # Add a point at the center
    plt.scatter([0], [0])
    x_label, y_label = getattr(env.observation_space, "names", ["x", "y"])
    final_show(save_figure, plot, directory, figure_name, x_label, y_label, title)


def plot_cartpole_critic_q(
    agent,
    env,
    env_string,
    directory,
    figure_name,
    plot=True,
    save_figure=True,
    action=None,
):
    """
    Visualization of the critic in a N-dimensional state space
    The N-dimensional state space is projected into its first two dimensions.
    A FeatureInverter wrapper should be used to select which features to put first to plot them
    :param agent: the critic agent to be plotted
    :param env: the environment
    :param env_string: the name of the environment
    :param plot: whether the plot should be interactive
    :param directory: the directory where to save the figure
    :param figure_name: the name of the file where to plot the function
    :param save_figure: whether the plot should be saved into a file
    :param action: the action for which we want to plot the value
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
            obs = obs.reshape(1, -1)
            obs = th.from_numpy(obs.astype(np.float32))

            if action is None:
                action = th.Tensor([0])
            value = agent.predict_value(obs[0], action)

            portrait[definition - (1 + index_y), index_x] = value.item()

    plt.figure(figsize=(10, 10))
    plt.imshow(
        portrait,
        cmap="inferno",
        extent=[state_min[0], state_max[0], state_min[2], state_max[2]],
        aspect="auto",
    )

    directory += "/" + env_string + "_critics/"
    title = env_string + " Critic"
    plt.colorbar(label="critic value")

    # Add a point at the center
    plt.scatter([0], [0])
    x_label, y_label = getattr(env.observation_space, "names", ["x", "y"])
    final_show(save_figure, plot, directory, figure_name, x_label, y_label, title)


def plot_lunarlander_critic_q(
    agent,
    env,
    env_string,
    directory,
    figure_name,
    plot=True,
    save_figure=True,
    action=None,
):
    """
    Visualization of the critic in a N-dimensional state space
    The N-dimensional state space is projected into its first two dimensions.
    A FeatureInverter wrapper should be used to select which features to put first to plot them
    :param agent: the critic agent to be plotted
    :param env: the environment
    :param env_string: the name of the environment
    :param plot: whether the plot should be interactive
    :param directory: the directory where to save the figure
    :param figure_name: the name of the file where to plot the function
    :param save_figure: whether the plot should be saved into a file
    :param action: the action for which we want to plot the value
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
            np.linspace(state_min[1], state_max[1], num=definition)
        ):
            obs = np.array([x])
            obs = np.append(obs, y)
            nb = range(len(state_min) - 2)
            for _ in nb:
                z = random.random() - 0.5
                obs = np.append(obs, z)
            obs = obs.reshape(1, -1)
            obs = th.from_numpy(obs.astype(np.float32))

            if action is None:
                action = th.Tensor([0])
            value = agent.predict_value(obs[0], action)

            portrait[definition - (1 + index_y), index_x] = value.item()

    plt.figure(figsize=(10, 10))
    plt.imshow(
        portrait,
        cmap="inferno",
        extent=[state_min[0], state_max[0], state_min[1], state_max[1]],
        aspect="auto",
    )

    directory += "/" + env_string + "_critics/"
    title = env_string + " Critic"
    plt.colorbar(label="critic value")

    # Add a point at the center
    plt.scatter([0], [0])
    x_label, y_label = getattr(env.observation_space, "names", ["x", "y"])
    final_show(save_figure, plot, directory, figure_name, x_label, y_label, title)


def plot_standard_critic_q(
    agent,
    env,
    env_string,
    directory,
    figure_name,
    plot=True,
    save_figure=True,
    action=None,
):
    """
    Visualization of the critic in a N-dimensional state space
    The N-dimensional state space is projected into its first two dimensions.
    A FeatureInverter wrapper should be used to select which features to put first to plot them
    :param agent: the critic agent to be plotted
    :param env: the environment
    :param env_string: the name of the environment
    :param plot: whether the plot should be interactive
    :param directory: the directory where to save the figure
    :param figure_name: the name of the file where to plot the function
    :param save_figure: whether the plot should be saved into a file
    :param action: the action for which we want to plot the value
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
            np.linspace(state_min[1], state_max[1], num=definition)
        ):
            obs = np.array([x])
            obs = np.append(obs, y)
            nb = range(len(state_min) - 2)
            for _ in nb:
                z = random.random() - 0.5
                obs = np.append(obs, z)
            obs = obs.reshape(1, -1)
            obs = th.from_numpy(obs.astype(np.float32))

            if action is None:
                action = th.Tensor([0])
            value = agent.predict_value(obs[0], action)

            portrait[definition - (1 + index_y), index_x] = value.item()

    plt.figure(figsize=(10, 10))
    plt.imshow(
        portrait,
        cmap="inferno",
        extent=[state_min[0], state_max[0], state_min[1], state_max[1]],
        aspect="auto",
    )

    directory += "/" + env_string + "_critics/"
    title = env_string + " Critic"
    plt.colorbar(label="critic value")

    # Add a point at the center
    plt.scatter([0], [0])
    x_label, y_label = getattr(env.observation_space, "names", ["x", "y"])
    final_show(save_figure, plot, directory, figure_name, x_label, y_label, title)
