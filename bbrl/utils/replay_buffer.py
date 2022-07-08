# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import copy
import torch

from bbrl.workspace import Workspace


class ReplayBuffer:
    def __init__(self, max_size, device=torch.device("cpu")):
        self.max_size = int(max_size)
        self.variables = None
        self.position = 0
        self.is_full = False
        self.device = device

    def init_workspace(self, all_tensors):
        """
        Create an array to stores workspace based on the given all_tensors keys.
        shape of stores tensors : [key] => [self.max_size][time_size][key_dim]
        Makes a copy of the input content
        """

        if self.variables is None:
            self.variables = {}
            for k, v in all_tensors.items():
                s = list(v.size())
                s[1] = self.max_size
                _s = copy.deepcopy(s)
                s[0] = _s[1]
                s[1] = _s[0]

                tensor = torch.zeros(*s, dtype=v.dtype, device=self.device)
                self.variables[k] = tensor
            self.is_full = False
            self.position = 0

    def _insert(self, k, indexes, v):
        self.variables[k][indexes] = v.detach().moveaxis((0, 1), (1, 0))

    def put(self, workspace):
        """
        Add a the content of a workspace to the replay buffer.
        The given workspace must have keys of shape : [time_size][batch_size][key_dim]
        """

        new_data = {
            k: workspace.get_full(k).detach().to(self.device) for k in workspace.keys()
        }
        self.init_workspace(new_data)

        batch_size = None
        arange = None
        indexes = None

        for k, v in new_data.items():
            if batch_size is None:
                batch_size = v.size()[1]
                # print(f"{k}: batch size : {batch_size}")
                # print("pos", self.position)
            if self.position + batch_size < self.max_size:
                # The case where the batch can be inserted before the end of the replay buffer
                if indexes is None:
                    indexes = torch.arange(batch_size) + self.position
                    arange = torch.arange(batch_size)
                    self.position = self.position + batch_size
                indexes = indexes.to(dtype=torch.long, device=v.device)
                arange = arange.to(dtype=torch.long, device=v.device)
                # print("insertion standard:", indexes)
                # # print("v shape", v.detach().shape)
                self._insert(k, indexes, v)
            else:
                # The case where the batch cannot be inserted before the end of the replay buffer
                # A part is at the end, the other part is in the beginning
                self.is_full = True
                # the number of data at the end of the RB
                batch_end_size = self.max_size - self.position
                # the number of data at the beginning of the RB
                batch_begin_size = batch_size - batch_end_size
                if indexes is None:
                    # print(f"{k}: batch size : {batch_size}")
                    # print("pos", self.position)
                    # the part of the indexes at the end of the RB
                    indexes = torch.arange(batch_end_size) + self.position
                    arange = torch.arange(batch_end_size)
                    # the part of the indexes at the beginning of the RB
                    # print("insertion intermediate computed:", indexes)
                    indexes = torch.cat((indexes, torch.arange(batch_begin_size)), 0)
                    arange = torch.cat((arange, torch.arange(batch_begin_size)), 0)
                    # print("insertion full:", indexes)
                    self.position = batch_begin_size
                indexes = indexes.to(dtype=torch.long, device=v.device)
                arange = arange.to(dtype=torch.long, device=v.device)
                self._insert(k, indexes, v)

    def size(self):
        if self.is_full:
            return self.max_size
        else:
            return self.position

    def print_obs(self):
        print(f"position: {self.position}")
        print(self.variables["env/env_obs"])

    def get_shuffled(self, batch_size):
        who = torch.randint(
            low=0, high=self.size(), size=(batch_size,), device=self.device
        )
        workspace = Workspace()
        for k in self.variables:
            workspace.set_full(k, self.variables[k][who].transpose(0, 1))

        return workspace

    def to(self, device):
        n_vars = {k: v.to(device) for k, v in self.variables.items()}
        self.variables = n_vars
