# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn

from bbrl.agents.agent import Agent


class Agents(Agent):
    """An agent that contains multiple agents that will be executed sequentially

    Args:
        Agent ([bbrl.Agent]): The agents
    """

    def __init__(self, *agents, name=None):
        """Creates the agent from multiple agents

        Args:
            name ([str], optional): [name of the resulting agent]. Defaults to None.
        """
        super().__init__(name=name)
        for a in agents:
            assert isinstance(a, Agent)
        self.agents = nn.ModuleList(agents)

    def __call__(self, workspace, **kwargs):
        for a in self.agents:
            a(workspace, **kwargs)

    def forward(self, **kwargs):
        raise NotImplementedError

    def seed(self, seed):
        for a in self.agents:
            a.seed(seed)

    def __getitem__(self, k):
        return self.agents[k]

    def get_by_name(self, n):
        r = []
        for a in self.agents:
            r = r + a.get_by_name(n)
        if n == self._name:
            r = r + [self]
        return r


class TemporalAgent(Agent):
    """Execute one Agent over multiple timesteps"""

    def __init__(self, agent, name=None):
        """The agent to transform to a temporal agent

        Args:
            agent ([bbrl.Agent]): The agent to encapsulate
            name ([str], optional): Name of the agent
        """
        super().__init__(name=name)
        self.agent = agent

    def __call__(self, workspace, t=0, n_steps=None, stop_variable=None, **kwargs):
        """Execute the agent starting at time t, for n_steps

        Args:
            workspace ([bbrl.Workspace]):
            t (int, optional): The starting timestep. Defaults to 0.
            n_steps ([type], optional): The number of steps. Defaults to None.
            stop_variable ([type], optional): if True everywhere (at time t), execution is stopped.
            Defaults to None = not used.
        """

        assert n_steps is not None or stop_variable is not None
        _t = t
        while True:
            self.agent(workspace, t=_t, **kwargs)
            if stop_variable is not None:
                s = workspace.get(stop_variable, _t)
                if s.all():
                    break
            _t += 1
            if n_steps is not None:
                if _t >= t + n_steps:
                    break

    def forward(self, **kwargs):
        raise NotImplementedError

    def seed(self, seed):
        self.agent.seed(seed)

    def get_by_name(self, n):
        r = self.agent.get_by_name(n)
        if n == self._name:
            r = r + [self]
        return r


class CopyTAgent(Agent):
    """An agent that copies a variable"""

    def __init__(self, input_name, output_name, detach=False, name=None):
        """
        Args:
        input_name ([str]): The variable to copy from
        output_name ([str]): The variable to copy to
        detach ([bool]): copy with detach if True
        """
        super().__init__(name=name)
        self.input_name = input_name
        self.output_name = output_name
        self.detach = detach

    def forward(self, t=None, **kwargs):
        """
        Args:
            t ([type], optional): if not None, copy at time t. Defaults to None.
        """
        if t is None:
            x = self.get(self.input_name)
            if not self.detach:
                self.set(self.output_name, x)
            else:
                self.set((self.output_name, t), x.detach())
        else:
            x = self.get((self.input_name, t))
            if not self.detach:
                self.set((self.output_name, t), x)
            else:
                self.set((self.output_name, t), x.detach())


class PrintAgent(Agent):
    """An agent to generate print in the console (mainly for debugging)
    It can be passed a list of strings corresponding to the variables to print
    or if nothing is passed, it prints all the existing variables in the workspace
    """

    def __init__(self, *names, name=None):
        """
        Args:
            names ([str], optional): The variables to print
        """
        super().__init__(name=name)
        self.names = names

    def reset(self):
        self.names = ()

    def forward(self, t, **kwargs):
        if self.names == ():
            self.names = self.workspace.keys()
        for n in self.names:
            try:
                value = self.get((n, t))
                print(n, " = ", value)
            except KeyError as e:
                print("Be sure to:")
                print(" — use the correct variable name / time step / workspace")
                print(
                    " — your print agent is called after the variable is written by another agent at the same time step"
                )
                if t == 0:
                    print(
                        f" — if you use r_t representation, your key {n} is not yet written at time t=0"
                    )
                raise KeyError("Variable ", n, " not found", e)


class EpisodesDone(Agent):
    """
    If done is encountered at time t, then done=True for all timeteps t'>=t
    It allows to simulate a single episode agent based on an autoreset agent
    """

    def __init__(self, in_var="env/done", out_var="env/done"):
        super().__init__()
        self.in_var = in_var
        self.out_var = out_var

    def forward(self, t, **kwargs):
        d = self.get((self.in_var, t))
        if t == 0:
            self.state = torch.zeros_like(d).bool()
        self.state = torch.logical_or(self.state, d)
        self.set((self.out_var, t), self.state)
