import torch
import torch.distributed as dist
import torch.nn as nn


class MinimalDDPFlat(nn.Module):

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

        with torch.no_grad():
            for param in module.parameters():
                dist.broadcast(param, src=0)
            for buffer in module.buffers():
                dist.broadcast(buffer, src=0)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        """Average gradients across ranks. Call after backward(), before step()."""
        # Collect all gradients that need synchronization
        grads = []
        for param in self.module.parameters():
            if param.requires_grad and param.grad is not None:
                grads.append(param.grad)

        if not grads:
            return

        # Flatten all gradients into a single contiguous tensor
        flat_grads = torch._utils._flatten_dense_tensors(grads)

        # Perform all-reduce on the flattened tensor
        dist.all_reduce(flat_grads, op=dist.ReduceOp.AVG, async_op=False)

        # Unflatten and copy back to original gradient tensors
        unflat_grads = torch._utils._unflatten_dense_tensors(flat_grads, grads)
        for grad, unflat_grad in zip(grads, unflat_grads):
            grad.copy_(unflat_grad)


def get_my_ddp(module: torch.nn.Module) -> torch.nn.Module:
    return MinimalDDPFlat(module)
