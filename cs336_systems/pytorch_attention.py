import sys
import torch
import torch.nn as nn
import timeit
import os
sys.path.insert(0, os.path.abspath('../cs336-basics'))

from cs336_basics.model import scaled_dot_product_attention
from cs336_basics.model import CausalMultiHeadSelfAttention

BATCH_SIZE = 8
NUM_ITERS = 20
WARMUP_ITERS = 10

def run_attention(d_model, seq_len, device):
    try:
        model = CausalMultiHeadSelfAttention(d_model, 1)
        model = torch.compile(model)
        model = model.to(device)
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
    finally:
        fname = "memory_snapshot-" + str(d_model) + "-" + str(seq_len) + ".pickle"
        torch.cuda.memory._dump_snapshot(fname)
        torch.cuda.memory._record_memory_history(enabled=None)

    return [x / (NUM_ITERS - WARMUP_ITERS) for x in [sum1, sum2, sum3, sum4]]

if __name__ == "__main__":
    d_models = [16, 32, 64, 128]
    seq_lens = [256, 1024, 4096, 8192, 16384]
    for d_model in d_models:
        for seq_len in seq_lens:
            try:
                val = run_attention(d_model, seq_len, 'cuda')
            except Exception as e:
                print(e)
                print(f"Some problem with ({d_model}, {seq_len})")
                continue

            print(d_model, seq_len, [round(1000 * x, 2) for x in val])



