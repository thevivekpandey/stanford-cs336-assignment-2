import torch
class FSDP:
    def __init__(self, 
                module: torch.nn.Module, 
                compute_dtype: torch.dtype | None = None):

        pass

    def forward(self, *inputs, **kwargs):
        pass

    def finish_gradient_synchronization(self):
        pass
