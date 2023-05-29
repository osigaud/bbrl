import sys
import os

import torch
from omegaconf import DictConfig

try:
    import gym
    import my_gym
    from bbrl.agents.gyma import AutoResetGymAgent
    no_gym = False
except ImportError:
    no_gym = True

from bbrl.workspace import Workspace
from bbrl.utils.replay_buffer import ReplayBuffer
from bbrl.agents import Agents, TemporalAgent, PrintAgent
from bbrl.agents.agent import Agent
from bbrl import get_class, get_arguments
import hydra

from bbrl.utils.chrono import Chrono

# HYDRA_FULL_ERROR = 1


class ActionAgent(Agent):
    # Create the action agent
    def __init__(self):
        super().__init__()

    def forward(self, t, **kwargs):
        action = torch.tensor([0])
        self.set(("action", t), action)


def make_gym_env(env_name):
    return gym.make(env_name)


def run_rb(cfg):

    train_env_agent = AutoResetGymAgent(
        get_class(cfg.gym_env),
        get_arguments(cfg.gym_env),
        cfg.algorithm.n_envs,
        cfg.algorithm.seed,
    )
    action_agent = ActionAgent()

    # Compose both previous agents
    tr_agent = Agents(train_env_agent, action_agent)

    # Get a temporal agent that can be executed in a workspace
    train_agent = TemporalAgent(tr_agent)

    train_workspace = Workspace()  # Used for training
    rb = ReplayBuffer(max_size=6)

    nb_steps = 0

    # 7) Training loop
    for epoch in range(cfg.algorithm.max_epochs):
        # Execute the agent in the workspace
        if epoch > 0:
            train_workspace.zero_grad()
            train_workspace.copy_n_last_steps(1)
            train_agent(
                train_workspace, t=1, n_steps=cfg.algorithm.n_steps - 1, stochastic=True
            )
        else:
            train_agent(
                train_workspace, t=0, n_steps=cfg.algorithm.n_steps, stochastic=True
            )

        nb_steps += cfg.algorithm.n_steps * cfg.algorithm.n_envs

        transition_workspace = train_workspace.get_transitions()

        obs = transition_workspace["env/env_obs"]
        print("obs ante:", obs)

        rb.put(transition_workspace)
        rb.print_obs()

        rb_workspace = rb.get_shuffled(cfg.algorithm.batch_size)

        obs = rb_workspace["env/env_obs"]
        print("obs post:", obs)


def main_loop(cfg):
    chrono = Chrono()
    logdir = "./plot/"
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    run_rb(cfg)
    chrono.stop()


@hydra.main(config_path="./configs/", config_name="rb_test.yaml", version_base="1.1")
def main(cfg: DictConfig):
    # print(OmegaConf.to_yaml(cfg))
    torch.manual_seed(cfg.algorithm.seed)
    main_loop(cfg)


if __name__ == "__main__":
    sys.path.append(os.getcwd())
    main()
