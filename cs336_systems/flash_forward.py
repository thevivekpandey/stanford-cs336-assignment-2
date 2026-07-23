import math
import torch
from einops import einsum

class MyAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        d_model = Q.shape[1]
        N_q = Q.shape[0]
        N_k = K.shape[0]
        #B_q, B_k = 16, 16
        B_q, B_k = 2, 2

        T_q = math.ceil(N_q / B_q)
        T_k = math.ceil(N_k / B_k)
        m_prev = torch.full((B_q, B_k), float('-inf'))
        m_next = torch.full((B_q, B_k), float('-inf'))
        print(m_prev)

        O = torch.zeros(d_model, d_model)
        for i in range(T_q):
            print(f"Outer loop {i}")
            Qi = Q[i * B_q:i * B_q + B_q, :]
            print(Qi)
            for j in range(T_k):
                print(f"Inner loop {j}")
                print(Kj)
                Kj = K[j * B_k:j * B_k + B_k, :]
                S = einsum(Qi, Kj, "... B_q d, ... B_k d -> ... B_q B_k") #/ torch.sqrt(torch.tensor([d_model]))
                print(S)
                rowmax = torch.max(S, dim=-1).values
                print(rowmax)
                m_curr[i, :] = torch.max(m_prev[i, :], rowmax)
                P = torch.exp(S - m_curr)

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError

if __name__ == "__main__":
    Q = torch.tensor([
                       [0.05, 0.10, 0.15, 0.20],
                       [0.25, 0.30, 0.35, 0.40],
                       [0.45, 0.50, 0.55, 0.60],
                       [0.65, 0.70, 0.75, 0.80]
                     ])

    K = torch.tensor([
                       [1.05, 1.10, 1.15, 1.20],
                       [1.25, 1.30, 1.35, 1.40],
                       [1.45, 1.50, 1.55, 1.60],
                       [1.65, 1.70, 1.75, 1.80]
                     ])

    V = torch.tensor([
                       [1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]
                     ])
    MyAttention.forward(1, Q, K, V)

