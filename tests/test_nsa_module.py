import torch

from nsa import selection_attention
from nsa.nsa import NSAAttention
from nsa.nsa_fused import NSAFusedAttention




bs, num_q_head, num_kv_head, head_dim = 1, 64, 4, 128
compress_block_size, compress_block_stride = 64, 16
selection_block, selected_block_count = 64, 32
seq_len = 1024*32

dtype = torch.bfloat16
device = "cuda"
torch.set_default_device(device)
torch.set_default_dtype(dtype)
torch.manual_seed(10)
q = torch.randn(bs*seq_len, num_q_head, head_dim, requires_grad=True)
k = torch.randn(bs*seq_len, num_kv_head, head_dim, requires_grad=True)
v = torch.randn(bs*seq_len, num_kv_head, head_dim, requires_grad=True)
t = torch.Tensor([0] + [seq_len] * bs)
cu_seq_len = torch.cumsum(t, dim=0).to(torch.int32).to(device)
torch.manual_seed(10)
attn = NSAAttention(head_dim, 0, True, None, 0, device=device, dtype=dtype)
torch.manual_seed(10)
fused_attn = NSAFusedAttention(head_dim, 0, True, None, 0, device=device, dtype=dtype)

o = attn(q, k, v, cu_seq_len, 0, causal=True)
fused_o = fused_attn(q, k, v, cu_seq_len, 0, causal=True)
assert not torch.isnan(o).any(), 'forward output has nan.'
assert not torch.isnan(fused_o).any(), 'forward output has nan.'

torch.testing.assert_close(o, fused_o, rtol=1e-2, atol=1e-2)
print('forward test passed.')
loss = (fused_o*fused_o).sum()
loss.backward()


# assert not torch.isnan(q.grad).any(), 'q.grad output has nan.'
# assert not torch.isnan(k.grad).any(), 'k.grad output has nan.'
# assert not torch.isnan(v.grad).any(), 'v.grad output has nan.'