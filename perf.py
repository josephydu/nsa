import torch

import triton
import time
from .tilelang import parallel_nsa as tilelang_nsa
torch.manual_seed(10)

dtype = torch.bfloat16
device = "cuda"
torch.set_default_device(device)
torch.set_default_dtype(dtype)
B, T, H, HQ, D, S, block_size, dtype = 1, 32, 1, 16, 32, 1, 32, torch.float16
torch.random.manual_seed(0)


q = torch.randn((B, T, HQ, D), dtype=dtype, device="cuda").requires_grad_(True)
k = torch.randn((B, T, H, D), dtype=dtype, device="cuda").requires_grad_(True)
v = torch.randn((B, T, H, D), dtype=dtype, device="cuda").requires_grad_(True)
g_slc = torch.ones((B, T, HQ), dtype=dtype, device="cuda").requires_grad_(True)
g_swa = torch.ones((B, T, HQ), dtype=dtype, device="cuda").requires_grad_(True)
do = torch.randn((B, T, HQ, D), dtype=dtype, device="cuda")


block_indices = torch.full((B, T, H, S), T, dtype=torch.long, device="cuda")
for b in range(B):
    for t in range(T):
        for h in range(H):
            i_i = torch.randperm(max(1, (t // block_size)))[:S]
            block_indices[b, t, h, :len(i_i)] = i_i
block_indices = block_indices.sort(-1)[0]

block_counts = torch.randint(1, S + 1, (B, T, H), device="cuda")


tilelang_nsa = lambda : tilelang_nsa(q, k, v, g_slc, g_swa, block_indices, block_size, block_counts)


def test(fwd_func):
    for i in range(5):
        o = fwd_func()
        loss = (o*o).sum()
        loss.backward()

    torch.cuda.synchronize()

    repeat=10
    forward_time = 0
    backward_time = 0
    for i in range(repeat):
        torch.cuda.synchronize()
        start = time.time()
        o = fwd_func()
        torch.cuda.synchronize()
        forward_time += time.time()-start
        loss = (o*o).sum()
        torch.cuda.synchronize()
        start = time.time()
        loss.backward()
        torch.cuda.synchronize()
        backward_time += time.time()-start
    
    forward_time /= repeat / 1e3
    backward_time /= repeat / 1e3

    total_time = forward_time + backward_time

    # d_model = num_q_head * head_dim
    # total_flops = 2 * 2 * (seq_len ** 2) * d_model * num_kv_head
    # tflops = total_flops / (total_time * 1e-3) / 1e12

    ms_per_iter = total_time
    print(f"Forward: {forward_time:.3f}ms | Backward: {backward_time:.3f}ms | Total: {total_time:.3f}ms")
    #print(f"Estimated TFLOPs/s: {tflops:.2f}")


test(tilelang_nsa)