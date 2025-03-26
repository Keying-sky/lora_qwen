def calculate_flops(N, H, D, L, V=151936, B=1, nsteps=1000, inference=True, infer_length=99):
    """
    Calculate FLOPS for Qwen2.5 model forward pass.
    
    Params:
    N : Context length (tokenwise)
    H : Number of attention heads
    D : Hidden dimension size
    L : Number of hidden layers
    V : Vocabulary size
    B : Batch size (for training mode)
    nsteps: Number of training steps (for training mode)
    inference : Whether in inference or training mode
    infer_length: length of focasted sequence (tokenwise), default=99 (equivalent to 10 timepoints)
    
    Return: Total FLOPS
    """
    # RoPE
    flops_rope = (4 * D**2 * N) / H - 2 * D * N
    
    # RMSNorm
    flops_rmsnorm = 2 * (3 * D * N + 10 * N)
    
    # self-Attention
    flops_self_attention = 6 * D**2 * N + 4 * D * N**2 + (22 * N**2 - N + 10) * H - D * N
    
    # FNN with SwiGLU
    flops_fnn = 28 * D * N + 16 * D**2 * N
    
    # one transformer layer
    flops_transformer = flops_rope + flops_rmsnorm + flops_self_attention + flops_fnn

    # output projection
    flops_output = (2 * D + 22) * V * N - V

    # full model forward pass (only one time)
    flops= L * flops_transformer + flops_output
    
    if not inference:
        flops = 3 * flops * B * nsteps    # training mode, add backward pass, batch size and nsteps
    else:
        flops = flops * infer_length      # inference mode, add forcasted length

    return flops
