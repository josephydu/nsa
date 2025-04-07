import torch

from nsa import selection_attention
from nsa.nsa import NSAAttention


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

forward_start = torch.cuda.Event(enable_timing=True)
forward_end = torch.cuda.Event(enable_timing=True)
backward_start = torch.cuda.Event(enable_timing=True)
backward_end = torch.cuda.Event(enable_timing=True)

forward_start.record()
o = attn(q, k, v, cu_seq_len, 0, causal=True)
forward_end.record()
assert not torch.isnan(o).any(), 'forward output has nan.'

loss = (o*o).sum()

backward_start.record()
loss.backward()
backward_end.record()

torch.cuda.synchronize()

forward_time = forward_start.elapsed_time(forward_end)
backward_time = backward_start.elapsed_time(backward_end)
total_time = forward_time + backward_time

d_model = num_q_head * head_dim
total_flops = 2 * 2 * (seq_len ** 2) * d_model * num_kv_head
tflops = total_flops / (total_time * 1e-3) / 1e12

print(f"Forward time: {forward_time:.2f}ms")
print(f"Backward time: {backward_time:.2f}ms")
print(f"Total time: {total_time:.2f}ms")
print(f"Estimated TFLOPs/s: {tflops:.2f}")
