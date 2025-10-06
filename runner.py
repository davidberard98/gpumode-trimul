import torch
from task import input_t, output_t
from contextlib import nullcontext
from impl import custom_kernel
import math

from triton.testing import do_bench

DeterministicContext = nullcontext


def make_match_reference(ref_kernel, rtol, atol):
    '''
    inp = generate_input(
        seqlen=32, bs=2, dim=64, hiddendim=128, seed=1024, nomask=False, distribution="normal"
    )
    '''
    inp = generate_input(
        seqlen=512, bs=2, dim=384, hiddendim=128, seed=1024, nomask=False, distribution="normal"
    )

    def get_clone():
        input_tensor, mask, weights, config = inp
        cloned_weights = {k: v.clone() for k, v in weights.items()}
        return (input_tensor.clone(), mask.clone(), cloned_weights, config)

    expected = ref_kernel(get_clone())

    def checker(cust_kern):
        actual = cust_kern(get_clone())
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)

        new_inp = get_clone()
        def fn():
            cust_kern(new_inp)
        ms = do_bench(fn)

        # Calculate GB based on all tensor sizes
        input_tensor, mask, weights, config = new_inp
        gb = (input_tensor.numel() * input_tensor.element_size() +
              mask.numel() * mask.element_size() +
              sum(w.numel() * w.element_size() for w in weights.values())) / 1e9
        print(f"ms: {ms}; GB/s: {gb/ms*1e3}")

        with torch.profiler.profile(record_shapes=True) as prof:
            for _ in range(5):
                fn()
                prof.step()

        torch.cuda.synchronize()
        prof.export_chrome_trace("trimul.json")

    return checker


def ref_kernel(data: input_t) -> output_t:
    """
    Reference implementation of TriMul using PyTorch.

    Args:
        data: Tuple of (input_tensor, mask, weights, config)

    Returns:
        Output tensor of shape [batch_size, seq_len, seq_len, dim]
    """
    with DeterministicContext():
        input_tensor, mask, weights, config = data

        from torch import nn, einsum

        # Extract configuration
        dim = config["dim"]
        hidden_dim = config["hidden_dim"]
        batch_size, seq_len, _, _ = input_tensor.shape

        # Layer normalization
        x = torch.nn.functional.layer_norm(
            input_tensor, (dim,), eps=1e-5,
            weight=weights['norm.weight'], bias=weights['norm.bias']
        )

        # Projections
        left = torch.nn.functional.linear(x, weights['left_proj.weight'])
        right = torch.nn.functional.linear(x, weights['right_proj.weight'])

        # Apply mask
        mask = mask.unsqueeze(-1)
        left = left * mask
        right = right * mask

        # Gates
        left_gate = torch.nn.functional.linear(x, weights['left_gate.weight']).sigmoid()
        right_gate = torch.nn.functional.linear(x, weights['right_gate.weight']).sigmoid()
        out_gate = torch.nn.functional.linear(x, weights['out_gate.weight']).sigmoid()

        # Apply gates
        left = left * left_gate
        right = right * right_gate

        # Triangle multiplication using einsum
        out = einsum('... i k d, ... j k d -> ... i j d', left, right)

        # Output normalization and projection
        out = torch.nn.functional.layer_norm(
            out, (hidden_dim,), eps=1e-5,
            weight=weights['to_out_norm.weight'], bias=weights['to_out_norm.bias']
        )
        out = out * out_gate
        out = torch.nn.functional.linear(out, weights['to_out.weight'])

        return out


def generate_input(
    seqlen: int,
    bs: int,
    dim: int,
    hiddendim: int,
    seed: int,
    nomask: bool,
    distribution: str,
) -> input_t:
    """
    Generates input for TriMul testing.

    Args:
        seqlen: Sequence length
        bs: Batch size
        dim: Feature dimension
        hiddendim: Hidden dimension
        seed: Random seed
        nomask: If True, use all-ones mask
        distribution: "normal" or "cauchy" for input distribution

    Returns:
        Tuple of (input_tensor, mask, weights, config)
    """
    batch_size = bs
    seq_len = seqlen
    hidden_dim = hiddendim
    no_mask = nomask

    config = {
        "hidden_dim": hidden_dim,
        "dim": dim,
    }

    gen = torch.Generator(device='cuda')
    gen.manual_seed(seed)

    # Generate input tensor based on distribution
    if distribution == "cauchy":
        input_tensor = torch.distributions.Cauchy(0, 2).sample(
            (batch_size, seq_len, seq_len, dim)
        ).to(device='cuda', dtype=torch.float32)
    else:  # normal distribution
        input_tensor = torch.randn(
            (batch_size, seq_len, seq_len, dim),
            device='cuda',
            dtype=torch.float32,
            generator=gen
        ).contiguous()

    if no_mask:
        mask = torch.ones(batch_size, seq_len, seq_len, device=input_tensor.device)
    else:
        mask = torch.randint(0, 2, (batch_size, seq_len, seq_len), device=input_tensor.device, generator=gen)

    # Initialize model weights
    weights = {}
    weights["norm.weight"] = torch.randn(dim, device="cuda", dtype=torch.float32)
    weights["norm.bias"] = torch.randn(dim, device="cuda", dtype=torch.float32)
    weights["left_proj.weight"] = torch.randn(hidden_dim, dim, device="cuda", dtype=torch.float32) / math.sqrt(hidden_dim)
    weights["right_proj.weight"] = torch.randn(hidden_dim, dim, device="cuda", dtype=torch.float32) / math.sqrt(hidden_dim)
    weights["left_gate.weight"] = torch.randn(hidden_dim, dim, device="cuda", dtype=torch.float32) / math.sqrt(hidden_dim)
    weights["right_gate.weight"] = torch.randn(hidden_dim, dim, device="cuda", dtype=torch.float32) / math.sqrt(hidden_dim)
    weights["out_gate.weight"] = torch.randn(hidden_dim, dim, device="cuda", dtype=torch.float32) / math.sqrt(hidden_dim)
    weights["to_out_norm.weight"] = torch.randn(hidden_dim, device="cuda", dtype=torch.float32)
    weights["to_out.weight"] = torch.randn(dim, hidden_dim, device="cuda", dtype=torch.float32) / math.sqrt(dim)
    weights["to_out_norm.bias"] = torch.randn(hidden_dim, device="cuda", dtype=torch.float32)

    return (input_tensor, mask, weights, config)


check_implementation = make_match_reference(ref_kernel, rtol=2e-2, atol=2e-2)

check_implementation(custom_kernel)
