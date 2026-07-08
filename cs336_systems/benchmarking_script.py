import sys
import os
import torch
# Add the parent directory to Python path
sys.path.insert(0, os.path.abspath('../cs336-basics'))

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW

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

def get_random_data():
    a = torch.randint(0, VOCAB_SIZE, size=(BATCH_SIZE, CONTEXT_LENGTH))
    b = torch.randint(0, VOCAB_SIZE, size=(BATCH_SIZE, CONTEXT_LENGTH))
    return a, b

def benchmarking_script():
    model_type = "small"
    print("Initing the model")
    model = BasicsTransformerLM(
        VOCAB_SIZE,
        CONTEXT_LENGTH,
        CONFIG[model_type]["d_model"],
        CONFIG[model_type]["num_layers"],
        CONFIG[model_type]["num_heads"],
        CONFIG[model_type]["d_ff"],
        10_000
    )
    print("Initing the optimizer")
    optimizer = AdamW(model.parameters())
    print("Getting data")
    i, t = get_random_data()
    print("Running forward pass")
    logits = model(i)
    print(logits)

if __name__ == "__main__":
    benchmarking_script()
