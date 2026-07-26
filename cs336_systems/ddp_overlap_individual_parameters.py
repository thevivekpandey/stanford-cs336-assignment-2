import torch
import torch.distributed as dist
import torch.nn as nn

class DDPOverlapIndividualParameters(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.works = []

        with torch.no_grad():
            for param in module.parameters():
                dist.broadcast(param, src=0)
            for buffer in module.buffers():
                dist.broadcast(buffer, src=0)

        self.handle = [
           p.register_post_accumulate_grad_hook(self._hook)
           for p in self.module.parameters() if p.requires_grad
        ]

    def _hook(self, param):
        self.works.append(
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, async_op=True)
        )
        
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        for w in self.works:
            w.wait()
        self.works.clear()


def get_my_ddp(module: torch.nn.Module) -> torch.nn.Module:
    return DDPOverlapIndividualParameters(module)
