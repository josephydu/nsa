import torch
from nsa.compression_kv import compress_kv

from torch.autograd import gradcheck

def test_compress_kv():
    # 配置测试参数
    batch_size = 2
    seq_len = 10  # 需要保证 (seq_len - block_size) % block_stride == 0
    num_heads = 4
    head_dim = 64
    block_stride = 2
    block_size = 3
    dtype = torch.bfloat16
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 构造输入数据
    torch.manual_seed(42)
    cu_seq_len = torch.tensor([0, 5, 10], dtype=torch.int32, device=device)
    
    # 初始化模型参数
    w_k = torch.randn(block_size*head_dim, head_dim, dtype=dtype, device=device, requires_grad=True)
    w_v = torch.randn(block_size*head_dim, head_dim, dtype=dtype, device=device, requires_grad=True)
    
    # 构造输入张量
    k = torch.randn(seq_len, num_heads, head_dim, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(seq_len, num_heads, head_dim, dtype=dtype, device=device, requires_grad=True)

    # 前向计算
    compressed_k, compressed_v = compress_kv(
        k, v, w_k, w_v, cu_seq_len, block_stride, block_size
    )

    print("前向计算验证:")
    print(f"压缩后的k形状: {compressed_k.shape}")
    print(f"压缩后的v形状: {compressed_v.shape}")
    print(f"压缩后的k示例值:\n{compressed_k.detach().cpu()[:,0,:5]}")
    print(f"压缩后的v示例值:\n{compressed_v.detach().cpu()[:,0,:5]}")

    # 反向传播测试
    def compress_func(k, v, w_k, w_v):
        return compress_kv(k, v, w_k, w_v, cu_seq_len, block_stride, block_size)

    # 梯度检查
    print("\n梯度检查:")
    test = gradcheck(compress_func, 
                    (k.float(), v.float(), w_k.float(), w_v.float()),  # gradcheck需要float精度
                    eps=1e-2,    
                    atol=1e-3,
                    rtol=1e-3,
                    nondet_tol=1e-3,  
                    check_undefined_grad=False,
                    raise_exception=False)
    print("梯度检查结果:", test)

    # 实际反向传播测试
    print("\n实际反向传播测试:")
    loss = compressed_k.sum() + compressed_v.sum()
    loss.backward()
    
    print("w_k梯度是否存在:", w_k.grad is not None)
    print("w_v梯度是否存在:", w_v.grad is not None)
    print("k梯度是否存在:", k.grad is not None)
    print("v梯度是否存在:", v.grad is not None)
    
    print("\nw_k梯度示例:")
    print(w_k.grad[0,:5])
    print("\nw_v梯度示例:")
    print(w_v.grad[0,:5])

if __name__ == "__main__":
    test_compress_kv()