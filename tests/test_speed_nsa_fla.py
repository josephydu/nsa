import torch

from nsa.nsa import NSAAttention
import triton

torch.manual_seed(10)

bs, num_q_head, num_kv_head, head_dim = 1, 64, 4, 128
compress_block_size, compress_block_stride = 64, 16
selection_block, selected_block_count = 64, 32
seq_len = 1024*32

dtype = torch.bfloat16
device = "cuda"
torch.set_default_device(device)
torch.set_default_dtype(dtype)

q = torch.randn(bs*seq_len, num_q_head, head_dim, requires_grad=True)
k = torch.randn(bs*seq_len, num_kv_head, head_dim, requires_grad=True)
v = torch.randn(bs*seq_len, num_kv_head, head_dim, requires_grad=True)
t = torch.Tensor([0] + [seq_len] * bs)
cu_seq_len = torch.cumsum(t, dim=0).to(torch.int32).to(device)

attn = NSAAttention(head_dim, 0, True, None, 0, device=device, dtype=dtype)

for _ in range(3):
    o = attn(q, k, v, cu_seq_len, 0, causal=True)
    loss = (o*o).sum()
    loss.backward()
    torch.cuda.synchronize()

forward_time = triton.testing.do_bench(
    lambda: attn(q, k, v, cu_seq_len, 0, causal=True),
    warmup=0,
    rep=40
)[0]

loss = (o.detach() * o.detach()).sum()  
grad_tensor = torch.ones_like(loss)
backward_time = triton.testing.do_bench(
    lambda: o.backward(grad_tensor, retain_graph=True),
    warmup=0,
    rep=40
)[0]
total_time = forward_time + backward_time

d_model = num_q_head * head_dim
total_flops = 2 * 2 * (seq_len ** 2) * d_model * num_kv_head
tflops = total_flops / (total_time * 1e-3) / 1e12

ms_per_iter = total_time
print(f"Forward: {forward_time:.3f}ms | Backward: {backward_time:.3f}ms | Total: {total_time:.3f}ms")
print(f"Estimated TFLOPs/s: {tflops:.2f}")
