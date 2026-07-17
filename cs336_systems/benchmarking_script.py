import sys
import os
import torch
import timeit
# Add the parent directory to Python path
sys.path.insert(0, os.path.abspath('../cs336-basics'))

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy

VOCAB_SIZE = 10000
BATCH_SIZE = 4
CONTEXT_LENGTH = 1024
CONFIG = {
    "test": {
        "d_model": 16,
        "d_ff": 24,
        "num_layers": 2,
        "num_heads": 2
    },
    "small": {
        "d_model": 768,
        "d_ff": 3072,
        "num_layers": 12,
        "num_heads": 12
    },
    "medium":{
        "d_model": 1024,
        "d_ff": 4096,
        "num_layers": 24,
        "num_heads": 16
    },
    "large": {
        "d_model": 1280,
        "d_ff": 5120,
        "num_layers": 36,
        "num_heads": 20
    },
    "xl": {
        "d_model": 2560,
        "d_ff": 10240,
        "num_layers": 32,
        "num_heads": 32
    },
    "10B": {
        "d_model": 4608,
        "d_ff": 12288,
        "num_layers": 50,
        "num_heads": 36
    }
}

def get_random_data(device: str):
    a = torch.randint(0, VOCAB_SIZE, size=(BATCH_SIZE, CONTEXT_LENGTH)).to(device)
    b = torch.randint(0, VOCAB_SIZE, size=(BATCH_SIZE, CONTEXT_LENGTH)).to(device)
    return a, b

def benchmarking_script(model_type):
    DEVICE = "cuda"
    model = BasicsTransformerLM(
        VOCAB_SIZE,
        CONTEXT_LENGTH,
        CONFIG[model_type]["d_model"],
        CONFIG[model_type]["num_layers"],
        CONFIG[model_type]["num_heads"],
        CONFIG[model_type]["d_ff"],
        10_000
    )
    model.to(DEVICE)

    optimizer = AdamW(model.parameters())
    model.train()
    for i in range(15):
        t1 = timeit.default_timer()

        a, b = get_random_data(DEVICE)
        torch.cuda.synchronize()

        t2 = timeit.default_timer()

        optimizer.zero_grad()
        torch.cuda.synchronize()

        t3 = timeit.default_timer()

        logits = model(a)
        torch.cuda.synchronize()

        t4 = timeit.default_timer()

        loss = cross_entropy(logits, b)
        torch.cuda.synchronize()

        t5 = timeit.default_timer()

        loss.backward()
        torch.cuda.synchronize()

        t6 = timeit.default_timer()
        optimizer.step()

        torch.cuda.synchronize()
        print(model_type, 
              i, 
              round(t2 - t1, 4), 
              round(t3 - t2, 4), 
              round(t4 - t3, 4), 
              round(t5 - t4, 4), 
              round(t6 - t5, 4))


if __name__ == "__main__":
    #benchmarking_script("test")
    #benchmarking_script("small")
    #benchmarking_script("medium")
    benchmarking_script("large")

    #benchmarking_script("xl")
    #benchmarking_script("10B")
