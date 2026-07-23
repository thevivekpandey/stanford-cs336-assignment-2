import sys
import math
import os
import time
import numpy as np
import argparse

import torch
import wandb

_A1 = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'stanford-cs336-assignment-1'))
sys.path.insert(0, _A1)                              # so `cs336_basics.X` resolves to assignment 1
sys.path.insert(1, os.path.join(_A1, 'cs336_basics'))  # so its modules' flat sibling imports resolve
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.rope import RoPE
from cs336_basics.adamw import AdamW, grad_norm
from cs336_basics.data_loading import data_loading
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.learning_rate_schedule import learning_rate_schedule
from cs336_basics.gradient_clipping import gradient_clipping
from ddp import get_my_ddp

import torch.distributed as dist
import torch.multiprocessing as mp

def set_process_group(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return device

def cleanup_process_group():
    dist.destroy_process_group()

class Train:
    def __init__(self,
                 train_path: str,
                 val_path: str,
                 d_model: int,
                 num_layers: int,
                 context_size: int,
                 batch_size: int,
                 num_steps: int,
                 vocab_size: int,
                 d_ff: int,
                 num_heads: int,
                 theta: float,
                 max_seq_len: int,
                 lr: float,
                 min_lr: float,
                 warmup_iter: int,
                 final_cosine_iter: int,
                 max_grad_norm: float,
                 max_train_seconds: float | None = None,
                 bytes_per_token: float = 4.365,
                 wandb_config=None,
                 wandb_group=None):
        self.train_path = train_path
        self.val_path = val_path
        self.d_model = d_model
        self.num_layers = num_layers
        self.context_size = context_size
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.vocab_size = vocab_size
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.lr = lr
        self.min_lr = min_lr
        self.warmup_iter = warmup_iter
        self.final_cosine_iter = final_cosine_iter
        self.max_grad_norm = max_grad_norm
        self.max_train_seconds = max_train_seconds
        self.bytes_per_token = bytes_per_token
        self.wandb_config = wandb_config
        self.wandb_group = wandb_group
        # loss is nats/token; /ln(2) -> bits/token; /bytes_per_token -> bits/byte,
        # which is comparable across tokenizers and is what the leaderboard scores.
        self.bpb_scale = 1.0 / (math.log(2) * bytes_per_token)

    def train(self, rank, world_size):
       self.device = set_process_group(rank, world_size)
       print(f"Starting training on rank {dist.get_rank()}")
       train_data = np.load(self.train_path, mmap_mode="r")
       val_data = np.load(self.val_path, mmap_mode="r")

       if rank == 0:
          wandb.init(
              entity="thevivekpandey-personal",
              project="stanford-cs336",
              name="test",
              group=self.wandb_group,
              config={
                  **self.wandb_config,
              },
          )

       print(f"train tokens: {len(train_data)}, val tokens: {len(val_data)}")

       rope = RoPE(self.theta, self.d_model // self.num_heads, self.max_seq_len, self.device)
       model = TransformerLM(self.vocab_size,
                             self.context_size,
                             self.d_model,
                             self.num_layers,
                             self.num_heads,
                             self.d_ff,
                             self.device,
                             rope)
       model = get_my_ddp(model)
       optimizer = AdamW(model.parameters(), lr=self.lr)
       DIR = "checkpoints"
       os.makedirs(DIR, exist_ok=True)
       start_time = time.time()
       step = 0
       while True:
          if self.max_train_seconds is not None and time.time() - start_time >= self.max_train_seconds:
              print(f"step {step}: reached time budget of {self.max_train_seconds}s; stopping")
              break
          if self.max_train_seconds is None and step >= self.num_steps:
              break
          model.train()

          lr = learning_rate_schedule(step,
                                      self.lr,
                                      self.min_lr,
                                      self.warmup_iter,
                                      self.final_cosine_iter)
          for group in optimizer.param_groups:
              group["lr"] = lr

          input_batch, target_batch = data_loading(train_data,
                                              self.batch_size,
                                              self.context_size,
                                              self.device)

          local_batch_size = self.batch_size // world_size
          start_idx = rank * local_batch_size
          end_idx = start_idx + local_batch_size
          local_input = input_batch[start_idx: end_idx]
          local_target = target_batch[start_idx: end_idx]

          optimizer.zero_grad()
          logits = model(local_input)
          loss = cross_entropy(logits, local_target)

          if rank == 0: #print loss only from rank 0
              print(f"step {step}: loss = {loss.item()}")

          loss.backward()

          model.finish_gradient_synchronization()
          norm = grad_norm(model.parameters())
          gradient_clipping(model.parameters(), self.max_grad_norm)
          optimizer.step()

          metrics = {
              "step": step,
              "train_loss": loss.item(),
              "train_bits_per_byte": loss.item() * self.bpb_scale,
              "learning_rate": optimizer.param_groups[0]["lr"],
              "grad_norm": norm.item(),
              "wall_clock_time": time.time() - start_time,
          }

          if step % 10 == 0:
              model.eval()
              with torch.no_grad():
                  input_batch, target_batch = data_loading(val_data,
                                                     self.batch_size,
                                                     self.context_size,
                                                     self.device)
                  logits = model(input_batch)
                  val_loss = cross_entropy(logits, target_batch)
              print(f"step {step}: val loss = {val_loss.item()}")
              metrics["val_loss"] = val_loss.item()
              metrics["val_bits_per_byte"] = val_loss.item() * self.bpb_scale

          if rank == 0:
              wandb.log(metrics, step=step)

          # ~5.6GB per checkpoint: 468M params of weights plus AdamW's two moment
          # buffers, all fp32. At 100k steps that is 50 checkpoints, ~281GB.
          if step % 2000 == 0:
              checkpoint_path = os.path.join(DIR, f"checkpoint_step_{step}.pt")
              torch.save({
                  "step": step,
                  # torch.compile wraps the model in an OptimizedModule, which
                  # prefixes every state_dict key with "_orig_mod.". Unwrap it so
                  # checkpoints load into a plain TransformerLM. getattr keeps
                  # this working if torch.compile is ever removed.
                  "model_state_dict": getattr(model, "_orig_mod", model).state_dict(),
                  "optimizer_state_dict": optimizer.state_dict(),
              }, checkpoint_path)
              print(f"step {step}: saved checkpoint to {checkpoint_path}")

          step += 1

       if rank == 0:
           wandb.finish()
       cleanup_process_group()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_model", type=int, default=512, required=False)
    parser.add_argument("--num_layers", type=int, default=16, required=False)
    parser.add_argument("--context_size", type=int, default=256, required=False)
    parser.add_argument("--batch_size", type=int, default=32, required=False)
    parser.add_argument("--num_steps", type=int, default=1000, required=False)
    parser.add_argument("--vocab_size", type=int, default=10000, required=False)
    parser.add_argument("--d_ff", type=int, default=3072, required=False)
    parser.add_argument("--num_heads", type=int, default=16, required=False)
    parser.add_argument("--theta", type=float, default=10000.0, required=False)
    parser.add_argument("--max_seq_len", type=int, default=1024, required=False)
    parser.add_argument("--lr", type=float, default=1e-3, required=False)
    parser.add_argument("--min_lr", type=float, default=1e-5, required=False)
    parser.add_argument("--max_grad_norm", type=float, default=1.0, required=False)
    parser.add_argument("--group", type=str, default=None, required=False)
    parser.add_argument("--max_train_seconds", type=float, default=None, required=False, help="If set, train for this many wall-clock seconds regardless of --num_steps")
    parser.add_argument("--train_data", type=str, default="TinyStoriesV2-GPT4-train.npy", required=False)
    parser.add_argument("--val_data", type=str, default="TinyStoriesV2-GPT4-val.npy", required=False)
    # Measured on the OWT text these .npy files came from. Loss per token is not
    # comparable across corpora with different compression, so log bits/byte too:
    # OWT is 4.365 bytes/token vs TinyStories' 3.24, which alone shifts per-token
    # loss upward without the model being any worse. Pass 3.24 for TinyStories.
    parser.add_argument("--bytes_per_token", type=float, default=4.365, required=False)
    parser.add_argument("--warmup_iter", type=int, default=2000, required=False)
    args = parser.parse_args()

    d_model = args.d_model
    num_layers = args.num_layers
    context_size = args.context_size
    batch_size = args.batch_size
    num_steps = args.num_steps
    vocab_size = args.vocab_size
    d_ff = args.d_ff
    num_heads = args.num_heads
    theta = args.theta
    max_seq_len = args.max_seq_len
    lr = args.lr
    min_lr = args.min_lr
    max_grad_norm = args.max_grad_norm

    warmup_iter = min(args.warmup_iter, num_steps)
    final_cosine_iter = num_steps

    val_data = np.load(args.val_data, mmap_mode="r")

    max_id = int(np.asarray(val_data[:1_000_000]).max())
    if max_id >= vocab_size:
        raise SystemExit(
            f"--vocab_size {vocab_size} is too small for {args.val_data} "
            f"(token id {max_id} present). The embedding would index out of bounds."
        )

    world_size = 2
    t = Train(args.train_data, args.val_data, d_model, num_layers, context_size, batch_size, num_steps, vocab_size, d_ff, num_heads, theta, max_seq_len, lr, min_lr, warmup_iter, final_cosine_iter, max_grad_norm, args.max_train_seconds, args.bytes_per_token, vars(args), args.group)
    mp.spawn(fn=t.train, args=(world_size,), nprocs=world_size, join=True) 
