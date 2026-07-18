import sys
import torch
import torch.nn as nn
import timeit
import os
sys.path.insert(0, os.path.abspath('../cs336-basics'))

from cs336_basics.model import scaled_dot_product_attention
from cs336_basics.model import CausalMultiHeadSelfAttention

BATCH_SIZE = 8
NUM_ITERS = 100
WARMUP_ITERS = 10

def run_attention(d_model, seq_len, device):
    model = CausalMultiHeadSelfAttention(d_model, 1)
    sum1, sum2, sum3, sum4 = 0.0, 0.0, 0.0, 0.0
    for i in range(NUM_ITERS):
        if i == WARMUP_ITERS:
            torch.cuda.memory._record_memory_history(max_entries=1_000_000)

        t1 = timeit.default_timer()
        x = torch.rand(BATCH_SIZE, seq_len, d_model, device=device)

        t2 = timeit.default_timer()
        y = model(x)
        torch.cuda.synchronize()

        t3 = timeit.default_timer()
        loss = y.sum()
        torch.cuda.synchronize()

        t4 = timeit.default_timer()
        loss.backward()
        torch.cuda.synchronize()

        t5 = timeit.default_timer()

        if i >= WARMUP_ITERS:
            sum1 += t2 - t1
            sum2 += t3 - t2
            sum3 += t4 - t3
            sum4 += t5 - t4

    torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
    torch.cuda.memory._record_memory_history(enabled=None)
    return sum1 / WARMUP_ITERS, sum2 / WARMUP_ITERS, sum3 / WARMUP_ITERS, sum4 / WARMUP_ITERS

if __name__ == "__main__":
    run_attention(16, 256, 'cpu')



