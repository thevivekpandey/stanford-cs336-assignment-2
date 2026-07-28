import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import Optimizer
from typing import Type, Any, List

class OptimizerStateSharding():

    def __init__(self,
                params,
                optimizer_cls: Type[Optimizer],
                **kwargs: Any):
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        self.all_params = list(params)
        param_assignments = self._assign_params_to_ranks(self.all_params)
        my_params = param_assignments[self.rank]
        self.optimizer = optimizer_cls(my_params, **kwargs)
        self.param_assignments = param_assignments

    def _assign_params_to_ranks(self, params: List[torch.nn.Parameter],
                                existing_assignments: List[List[torch.nn.Parameter]] = None) -> List[List[torch.nn.Parameter]]:
        if existing_assignments is None:
            assignments = [[] for _ in range(self.world_size)]
        else:
            assignments = existing_assignments

        rank_element_counts = [0] * self.world_size
        for rank in range(self.world_size):
            for param in assignments[rank]:
                rank_element_counts[rank] += param.numel()

        for param in params:
            if not param.requires_grad:
                continue

            num_elements = param.numel()
            min_rank = min(range(self.world_size), key=lambda r: rank_element_counts[r])

            assignments[min_rank].append(param)
            rank_element_counts[min_rank] += num_elements

        return assignments

    def step(self, closure=None, **kwargs):
        self.optimizer.step(closure, **kwargs)

        # All ranks must participate in all broadcasts in the same order
        for rank in range(self.world_size):
            for param in self.param_assignments[rank]:
                dist.broadcast(param.data, src=rank)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def add_param_group(self, param_group: dict[str, Any]):
        new_params = param_group['params']
        if not isinstance(new_params, list):
            new_params = list(new_params)

        self.all_params.extend(new_params)

        self.param_assignments = self._assign_params_to_ranks(new_params, self.param_assignments)

        my_new_params = [p for p in new_params if p in self.param_assignments[self.rank]]

        if my_new_params:
            my_param_group = {k: v for k, v in param_group.items() if k != 'params'}
            my_param_group['params'] = my_new_params

            self.optimizer.add_param_group(my_param_group)
