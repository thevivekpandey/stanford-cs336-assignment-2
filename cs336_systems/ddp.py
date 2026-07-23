import torch
import torch.distributed as dist
import torch.nn as nn


class NaiveDDP(nn.Module):
    """Minimal DDP (assignment section 5.2).

    Every rank starts from rank 0's parameters, then each rank runs forward and
    backward on its own shard of the batch. Gradients are all-reduced *after*
    the backward pass has fully finished, one collective per parameter tensor.
    Both of those are the limitations that section 5.3 goes on to fix, so they
    are deliberate here: this is the baseline the later versions are measured
    against.
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

        with torch.no_grad():
            for param in module.parameters():
                dist.broadcast(param, src=0)
            # Buffers (e.g. RoPE caches, running stats) are part of the model
            # state too, so they have to start out identical as well.
            for buffer in module.buffers():
                dist.broadcast(buffer, src=0)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        """Average gradients across ranks. Call after backward(), before step()."""
        for param in self.module.parameters():
            if param.requires_grad and param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, async_op=False)


def get_my_ddp(module: torch.nn.Module) -> torch.nn.Module:
    return NaiveDDP(module)
