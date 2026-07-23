import os
import torch
import timeit
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def distributed_demo(rank, world_size):
    setup(rank, world_size)
    size = 10_000_000_000

    for i in range(20):
        data = torch.rand(size // 4, dtype=torch.float32,device=f"cuda:{rank}")
        #print(f"rank {rank} data (before all-reduce): {data}")
        t1 = timeit.default_timer()
        dist.all_reduce(data, async_op=False)
        torch.cuda.synchronize()
        t2 = timeit.default_timer()
        print(f"rank {rank}: {round(1000 * (t2  - t1), 2)}")
        #print(f"rank {rank} data (after all-reduce): {data}")
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 4
    mp.spawn(fn=distributed_demo, args=(world_size, ), nprocs=world_size, join=True)
