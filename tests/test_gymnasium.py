import copy
from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Tuple
import numpy as np
import pytest

import torch
import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium import spaces
from bbrl.agents import Agent, TemporalAgent, Agents
from bbrl.workspace import Workspace
from bbrl.agents.gymnasium import ParallelGymAgent, VecGymAgent, make_env
from gymnasium.wrappers import Autoreset
from gymnasium.vector import SyncVectorEnv


class MyEnv(gym.Env):
    """Simple test environment

    action 1: 0 -> 1 -> 2
    action 0: 1 -> 0 -> 2

    Target (reward=5) is 2, max 5 steps
    Each action is -1 reward
    """

    MOVES = {0: [2, 1], 1: [0, 2], 2: [2, 2]}
    TARGET = 2

    def __init__(self):
        self._max_steps = 5
        self.observation_space = spaces.Box(0, len(MyEnv.MOVES), shape=(1,), dtype=int)
        self.action_space = spaces.Discrete(2)

    def _get_obs(self):
        return self._agent_location

    def _get_info(self):
        return {}

    def reset(self, seed=0, options={}):
        self._agent_location = 0
        self._step = 0
        return self._get_obs(), self._get_info()

    def step(self, action):
        # Move
        self._agent_location = MyEnv.MOVES[self._agent_location][action]
        self._step += 1

        # An episode is done iff the agent has reached the target
        terminated = self._agent_location == MyEnv.TARGET
        reward = 5 if terminated else -1
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, self._step >= self._max_steps, info


class ActorAgent(Agent):
    def __init__(self, *actions):
        super().__init__()
        self.actions = torch.IntTensor(actions).t().reshape(-1, len(actions))

    def forward(self, t):
        self.set(("action", t), self.actions[t])


# TODO: VecGymAgent
def test_gymnasium_agent():
    make_env_fn = MyEnv

    env = ParallelGymAgent(make_env_fn, 2)
    actor = ActorAgent([1, 1], [0, 1])
    workspace = Workspace()

    env(workspace, t=0)
    actor(workspace, t=0)
    assert (workspace["env/env_obs"] == torch.Tensor([0, 0])).all()

    env(workspace, t=1)
    assert (workspace["env/env_obs"] == torch.Tensor([[0, 0], [1, 2]])).all()


def check(
    workspace,
    scenarios,
):
    for ix, scenario in enumerate(scenarios):
        for key, value in scenario.items():
            assert (
                workspace[key][:, ix] == torch.Tensor(value)
            ).all(), f"Error with scenario {ix} / {key}"


SCENARIOS = []


def add_scenarios(autoreset: bool, include_last_state: bool, scenarios: List[Dict]):
    # Add two scenarios (reward at t or t+1)
    n_steps = None
    for ix, scenario in enumerate(scenarios):
        for key, value in scenario.items():
            if n_steps is None:
                n_steps = len(value)
            else:
                assert len(value) == n_steps

        SCENARIOS.append(
            [autoreset, include_last_state, False, copy.deepcopy(scenarios)]
        )

        _scenarios = copy.deepcopy(scenarios)
        for scenario in _scenarios:
            scenario["env/cumulated_reward"] = []
            cumulated_reward = 0
            for reward, done in zip(scenario["env/reward"], scenario["env/done"]):
                cumulated_reward += reward
                scenario["env/cumulated_reward"].append(cumulated_reward)
                if done and autoreset:
                    cumulated_reward = 0

            scenario["env/reward"] = scenario["env/reward"][1:]
            scenario["env/reward"].append(0)
            scenario["env/cumulated_reward"] = scenario["env/cumulated_reward"][1:]
            scenario["env/cumulated_reward"].append(
                scenario["env/cumulated_reward"][-1]
            )
        SCENARIOS.append([autoreset, include_last_state, True, _scenarios])


add_scenarios(
    True,
    False,
    [
        # 0,2 / 0,1,2 / 0
        {
            "env/env_obs": [0, 0, 1, 0],
            "action": [0, 1, 1, 1],
            "env/done": [False, True, False, True],
            "env/terminated": [False, True, False, True],
            "env/truncated": [False, False, False, False],
            "env/reward": [0, 5, -1, 5],
        }
    ],
)

add_scenarios(
    True,
    True,
    [
        # 0,2 / 0,1,2 / 0
        {
            "env/env_obs": [0, 2, 0, 1, 2, 0],
            "action": [0, 0, 1, 1, 1, 0],
            "env/done": [False, True, False, False, True, False],
            "env/terminated": [False, True, False, False, True, False],
            "env/truncated": [False, False, False, False, False, False],
            "env/reward": [0, 5, 0, -1, 5, 0],
        }
    ],
)

# Time-out
add_scenarios(
    True,
    False,
    [
        {
            "env/env_obs": [0, 1, 0, 1, 0, 0],
            "action": [1, 0, 1, 0, 1, 0],
            "env/done": [False, False, False, False, False, True],
            "env/terminated": [False, False, False, False, False, False],
            "env/truncated": [False, False, False, False, False, True],
            "env/reward": [0, -1, -1, -1, -1, -1],
        }
    ],
)

# No autoreset
add_scenarios(
    False,
    False,
    [
        {
            "env/env_obs": [0, 1, 2, 2],
            "action": [1, 1, 1, 1],
            "env/done": [False, False, True, True],
            "env/terminated": [False, False, True, True],
            "env/truncated": [False, False, False, False],
            "env/reward": [0, -1, 5, 0],
        }
    ],
)


@pytest.mark.parametrize(
    "autoreset,include_last_state,reward_at_t,scenarios", SCENARIOS
)
def test_gymnasium_autoreset(autoreset, include_last_state, reward_at_t, scenarios):
    make_env_fn = (
        (lambda: gym.Wrapper(Autoreset(MyEnv())))
        if autoreset
        else (lambda: MyEnv())
    )

    n_steps = len(scenarios[0]["action"])

    env = ParallelGymAgent(
        make_env_fn,
        len(scenarios),
        reward_at_t=reward_at_t,
        include_last_state=include_last_state,
    )
    actor = ActorAgent(*(scenario["action"] for scenario in scenarios))
    agents = Agents(env, actor)
    t_agents = TemporalAgent(agents)

    workspace = Workspace()
    t_agents(workspace, n_steps=n_steps)

    check(workspace, scenarios)


class DeterministicEnv(gym.Env):
    """A deterministic environment for testing VecGymAgent vs ParallelGymAgent.

    The environment has a 1D continuous observation space and discrete actions.
    State transitions are deterministic based on seed and actions.
    Episode terminates after 10 steps or if state > 2.
    """

    def __init__(self):
        self.observation_space = spaces.Box(low=-10, high=10, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self._max_steps = 10
        self._state = None
        self._step_count = 0
        self._rng = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Use seeded RNG for deterministic but seed-dependent behavior
        self._rng = np.random.default_rng(seed)
        initial_val = self._rng.uniform(-1, 1)
        self._state = np.array([initial_val, 0.0], dtype=np.float32)
        self._step_count = 0
        return self._state.copy(), {}

    def step(self, action):
        self._state[0] += (action - 1) * 0.1
        self._state[1] = self._step_count
        self._step_count += 1

        terminated = self._state[0] > 2.0
        truncated = self._step_count >= self._max_steps
        reward = 1.0 if terminated else -0.1

        return self._state.copy(), reward, terminated, truncated, {}


@pytest.mark.parametrize("num_envs", [1, 2, 4])
@pytest.mark.parametrize("n_steps", [3, 5, 10])
@pytest.mark.parametrize("reward_at_t", [False, True])
def test_vec_gym_agent_matches_parallel(num_envs, n_steps, reward_at_t):
    """Test that VecGymAgent produces the same outputs as ParallelGymAgent."""
    seed = 42

    # Create ParallelGymAgent with autoreset (VecGymAgent always uses autoreset)
    def make_env_parallel():
        return Autoreset(DeterministicEnv())

    parallel_env = ParallelGymAgent(
        make_env_parallel,
        num_envs,
        reward_at_t=reward_at_t,
        include_last_state=False,  # VecGymAgent doesn't include last state
    )
    parallel_env.seed(seed)

    # Create VecGymAgent
    def make_vec_env():
        return SyncVectorEnv([lambda: DeterministicEnv() for _ in range(num_envs)])

    vec_env = VecGymAgent(
        make_vec_env,
        reward_at_t=reward_at_t,
    )
    vec_env.seed(seed)

    # Create a deterministic actor that chooses action based on timestep
    class DeterministicActor(Agent):
        def __init__(self, num_envs):
            super().__init__()
            self.num_envs = num_envs

        def forward(self, t):
            # Deterministic action: cycle through 0, 1, 2
            actions = torch.tensor([(t + i) % 3 for i in range(self.num_envs)])
            self.set(("action", t), actions)

    actor = DeterministicActor(num_envs)

    # Run ParallelGymAgent
    parallel_workspace = Workspace()
    parallel_agents = Agents(parallel_env, actor)
    parallel_t_agents = TemporalAgent(parallel_agents)
    parallel_t_agents(parallel_workspace, n_steps=n_steps)

    # Run VecGymAgent
    vec_workspace = Workspace()
    vec_agents = Agents(vec_env, actor)
    vec_t_agents = TemporalAgent(vec_agents)
    vec_t_agents(vec_workspace, n_steps=n_steps)

    # Compare outputs (excluding timestep which VecGymAgent doesn't provide)
    keys_to_compare = ["env/env_obs", "env/reward", "env/done", "env/terminated", "env/truncated"]

    for key in keys_to_compare:
        parallel_val = parallel_workspace[key]
        vec_val = vec_workspace[key]
        assert parallel_val.shape == vec_val.shape, (
            f"Shape mismatch for {key}: {parallel_val.shape} vs {vec_val.shape}"
        )
        assert torch.allclose(parallel_val.float(), vec_val.float(), atol=1e-5), (
            f"Value mismatch for {key}:\nParallel:\n{parallel_val}\nVec:\n{vec_val}"
        )
