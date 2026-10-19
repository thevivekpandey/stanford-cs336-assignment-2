import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import Optimizer
from typing import Type, Any, List

class OptimizerStateSharding(Optimizer):

    def __init__(self,
                params,
                optimizer_cls: Type[Optimizer],
                **kwargs: Any):
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = kwargs
        # Built lazily: the super-class constructor below calls add_param_group(),
        # and that is where we first learn which parameters belong to this rank.
        self.optimizer = None

        self.all_params = []
        self.param_assignments = [[] for _ in range(self.world_size)]
        # self.param_groups[i] is optimized by self.optimizer.param_groups[j] where
        # j = self._inner_group_idx[i]; None when this rank owns nothing in group i.
        self._inner_group_idx = []

        # Normalizes `params` (handles both a plain iterable of parameters and a
        # list of param-group dicts), then calls self.add_param_group() per group.
        # kwargs become the defaults, so self.param_groups carries lr, betas, etc.
        super().__init__(params, kwargs)

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

    def _sync_hyperparameters(self):
        # Hyperparameters are set on our own param_groups (e.g. an LR schedule doing
        # `for g in optimizer.param_groups: g["lr"] = lr`), but the wrapped optimizer
        # holds separate group dicts, so push the values down before stepping.
        for outer_idx, inner_idx in enumerate(self._inner_group_idx):
            if inner_idx is None:
                continue
            inner_group = self.optimizer.param_groups[inner_idx]
            for key, value in self.param_groups[outer_idx].items():
                if key != 'params':
                    inner_group[key] = value

    def step(self, closure=None, **kwargs):
        loss = None
        if self.optimizer is not None:
            self._sync_hyperparameters()
            loss = self.optimizer.step(closure, **kwargs)
        elif closure is not None:
            loss = closure()

        # All ranks must participate in all broadcasts in the same order
        for rank in range(self.world_size):
            for param in self.param_assignments[rank]:
                dist.broadcast(param.data, src=rank)

        return loss

    def zero_grad(self, set_to_none: bool = True):
        # Clear every parameter's grad, not just this rank's shard: backward()
        # knows nothing about ownership and accumulates into p.grad for the whole
        # model on every rank. Grads for params owned by other ranks would
        # otherwise grow without bound and corrupt anything that reads gradients
        # globally, e.g. the global norm in gradient_clipping(). self.param_groups
        # covers the full, unsharded model, so the super-class does the right thing.
        super().zero_grad(set_to_none=set_to_none)

    def add_param_group(self, param_group: dict[str, Any]):
        # Register the group on the super-class first: it validates the parameters,
        # turns 'params' into a list, and fills in any hyperparameter missing from
        # the group using self.defaults. self.param_groups therefore always
        # describes the full, unsharded model.
        super().add_param_group(param_group)
        registered_group = self.param_groups[-1]
        new_params = registered_group['params']

        self.all_params.extend(new_params)

        self.param_assignments = self._assign_params_to_ranks(new_params, self.param_assignments)

        # Compare by identity: `p in <list of tensors>` invokes Tensor.__eq__ and
        # blows up with "Boolean value of Tensor ... is ambiguous".
        my_param_ids = {id(p) for p in self.param_assignments[self.rank]}
        my_new_params = [p for p in new_params if id(p) in my_param_ids]

        if not my_new_params:
            self._inner_group_idx.append(None)
            return

        my_param_group = {k: v for k, v in registered_group.items() if k != 'params'}
        my_param_group['params'] = my_new_params

        if self.optimizer is None:
            self.optimizer = self.optimizer_cls([my_param_group], **self.optimizer_kwargs)
        else:
            self.optimizer.add_param_group(my_param_group)

        self._inner_group_idx.append(len(self.optimizer.param_groups) - 1)
