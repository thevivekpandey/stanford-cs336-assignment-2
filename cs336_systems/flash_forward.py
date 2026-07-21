import math
import torch
import einsum

class MyAttention(torch.autograd.function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        d_model = Q.shape[1]
        N_q = Q.shape[0]
        N_k = K.shape[0]
        B_q, B_k = 16, 16

        T_q = math.ceil(N_q / B_q)
        T_k = math.ceil(N_k / B_k)
        m_prev = torch.zeros(B_q, B_k)
        m_next = torch.zeros(B_q, B_k)

        O = torch.zeros(d_model, d_model)
        for i in range(T_q - 1):
            start_q = i * B_q
            Qi = Q[i * B_q:i * B_q + B_q, :]
            for j in range(T_k - 1):
                Kj = K[j * B_k:j * B_k + B_k, :]
                S = einsum(Qi, Kj, "... B_q d, ... B_k d -> ... B_q B_k") / torch.sqrt(d_model)
                rowmax = torch.max(S, dim=-1)
                m_curr = 





    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError


