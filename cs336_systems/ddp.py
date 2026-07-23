import torch
import torch.distributed as dist
import torch.nn as nn

class NaiveDDP(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
    
        for param in module.parameters():
            dist.broadcast(param.data, src=0)

        for param in module.parameters():
            if param.requires_grad:
                param.register_hook(self._gradient_hook)

    def _gradient_hook(self, grad):
        dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=False)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        pass

def get_my_ddp(module: torch.nn.Module) -> torch.nn.Module:
    return NaiveDDP(module)
