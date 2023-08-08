# coding=utf-8
#
# Copyright © Sorbonne University
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from abc import ABC
from typing import Optional

from bbrl.agents import Agent


class SeedableAgent(Agent, ABC):
    """
    `SeedableAgent` is used as a convention to represent agents that
    are seeded (not mandatory)
    """

    def __init__(self, seed: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self._seed = seed

    def seed(self, seed: int):
        """Provide a seed to this agent. Useful if the agent is stochastic.

        Args:
            seed (int): The seed to use
        """
        if self._seed is not None:
            raise Exception(
                "Your {self} agent is already seeded with {seed}. You cannot seed it again with {new_seed}\n"
                "If you want to seed it again, une one of SAgentLast, SAgentMean, SAgentSum".format(
                    self=self.__class__.__name__,
                    seed=self._seed,
                    new_seed=seed,
                )
            )
        self._seed = seed
        return self


class SeedableAgentLast(SeedableAgent, ABC):
    """
    `SeedableAgentLast` is used as a convention to represent agents that
    are seeded (not mandatory) and that use the last seed provided."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def seed(self, seed: int):
        """Provide a seed to this agent. Useful is the agent is stochastic.

        Args:
            seed (int): The seed to use
        """
        self._seed = seed


class SeedableAgentSum(SeedableAgent, ABC):
    """
    `SeedableAgentSum` is used as a convention to represent agents that
    are seeded (not mandatory) and that use the sum of all seeds provided.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._seeds = []

    def seed(self, seed: int) -> None:
        """Provide a seed to this agent. Useful is the agent is stochastic.

        Args:
            seed (int): The seed to use
        """
        self._seeds.append(seed)
        self._seed = sum(self._seeds)


class SeedableAgentMean(SeedableAgentSum, ABC):
    """
    `SeedableAgentMean` is used as a convention to represent agents that
    are seeded (not mandatory) and that use the mean of all seeds provided.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def seed(self, seed: int):
        """Provide a seed to this agent. Useful is the agent is stochastic.

        Args:
            seed (int): The seed to use
        """
        super().seed(seed)
        self._seed = int(sum(self._seeds) / len(self._seeds))
