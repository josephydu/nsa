import torch

from nsa.nsa_fused import NSAFusedAttention
import triton
import time
from flash_attn.flash_attn_interface import flash_attn_varlen_func
from torch.profiler import profile, record_function, ProfilerActivity
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


attn = NSAFusedAttention(head_dim, 0, True, None, 0, device=device, dtype=dtype)

nsa_fwd = lambda : attn(q, k, v, cu_seq_len, 0, causal=True)
flash_fwd = lambda : flash_attn_varlen_func(q, k, v, cu_seq_len, cu_seq_len, seq_len, seq_len)


def test(fwd_func, name):
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
    print('********* ',name, ' ***********')
    print(f"Forward: {forward_time:.3f}ms | Backward: {backward_time:.3f}ms | Total: {total_time:.3f}ms")
    sort_by_keyword = device + "_time_total"
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        o = fwd_func()
    print('Forward profile')
    print(prof.key_averages().table(sort_by=sort_by_keyword, row_limit=10))
    loss = (o*o).sum()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        loss.backward()
    print('Backward profile')
    print(prof.key_averages().table(sort_by=sort_by_keyword, row_limit=10))


# def trace_handler(prof: torch.profiler.profile):
#     file_prefix = f"nsa"

#     # Construct the trace file.
#     prof.export_chrome_trace(f"{file_prefix}.json.gz")

#     # Construct the memory timeline file.
#     prof.export_memory_timeline(f"{file_prefix}.html", device="cuda:0")

# test(nsa_fwd, 'NSA')
# # test(flash_fwd, 'FLASH ATTN')
# with torch.profiler.profile(
#        activities=[
#            torch.profiler.ProfilerActivity.CPU,
#            torch.profiler.ProfilerActivity.CUDA,
#        ],
#        schedule=torch.profiler.schedule(wait=0, warmup=0, active=6, repeat=1),
#        record_shapes=True,
#        profile_memory=True,
#        with_stack=True,
#        on_trace_ready=trace_handler,
#    ) as prof:
#     with record_function("## forward ##"):
#         o = nsa_fwd()
#         loss = (o*o).sum()
#     with record_function("## backward ##"):
#         loss.backward()


torch.cuda.memory._record_memory_history(
    max_entries=10000
)
with record_function("## forward ##"):
    o = nsa_fwd()
loss = (o*o).sum()
with record_function("## backward ##"):
    loss.backward()

torch.cuda.memory._dump_snapshot(f"nsa.pickle")
torch.cuda.memory._record_memory_history(enabled=None)



# test(nsa_fwd)
# test(flash_fwd)