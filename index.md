---
layout: post
title: "GPU Mode: Trimul"
date: 2025-10-02
author: davidberard98
categories: [gpu, optimization, cuda, triton]
---

# GPU Mode: Trimul

**David Berard** | October 2, 2025

---

I spent a day or two working on the [GPU Mode kernel challenge for the Trimul operator](https://tinyurl.com/gpumode-trimul). This challenge involves optimizing a key operation from AlphaFold3 that processes pairwise sequence embeddings—tensors with shape `[batch, seq_len, seq_len, hidden_dim]`.

Here's a quick overview of my solution and how I approached the problem. As a disclaimer, I didn't record most of this during the contest; this is best-effort recollection of the steps I took using re-creations of the approximate state of the code to get performance numbers after each step.

**Huge thanks to the GPU mode team and the problem authors (Mark Saroufim, Matej Sirovatka, Alex Zhang, and I'm sure many other people I haven't mentioned)**. It was a super fun project and I wish I had a bit more time to look more deeply at the problem! And congrats to [Arseni Ivanov](https://arseniivanov.github.io/blog.html) who won the contest for A100 and MI300!

## The Problem: Outgoing TriMul Kernel

The challenge involves optimizing a "triangle multiplication" operator used in AlphaFold3. For full problem details, see the [challenge description](https://tinyurl.com/gpumode-trimul).

## Setup / Testing

I don't have a local GPU, so I rented compute. I spent a total of about $20, mostly on vast.ai compute. For the most part I rented a 5070-5090 just for local testing to make sure that my kernels were functional, and occasionally I would rent an A100 (\~$0.70/hr), H100 (\~$2/hr), or B200 (\~$8/hr) for performance tuning and profiling.

I was mostly targeting H100 because it is the platform I am most familiar with.

## Strategy

As stated in the problem remarks, it's quite difficult to fuse this into a single kernel efficiently. The layernorm reductions encourage certain blocking strategies, while the matmuls encourage others.

So I decided to skip the "perfect fusion" approach and start with easier, incremental optimizations.

**Testing configuration:** For most of my local testing and the performance numbers below, I used `seq_len=512`, `batch_size=2`, `dim=384`, `hidden_dim=128` on H100. Note - these numbers are different from the geomean numbers reported on the leaderboard (1088us for B200, 1371us for H100)

### 0. Functional Implementation

**Baseline performance:** 8.61ms

I started by making the implementation functional instead of using an nn.Module, to simplify the process of replacing operations with fused/custom versions.

I also [enabled tf32](https://docs.pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices) to enable tensor cores.

### 1. Fuse the Linear Computations

The original code performs many redundant loads and stores in this section:

```python
# Compute left and right paths
left = self.left_proj(x)
right = self.right_proj(x)

mask = mask.unsqueeze(-1)
left = left * mask
right = right * mask

left_gate = self.left_gate(x).sigmoid()
right_gate = self.right_gate(x).sigmoid()
out_gate = self.out_gate(x).sigmoid()

left = left * left_gate
right = right * right_gate
```

I wrote a **Triton persistent matmul kernel (with TMA)** which implements a matmul loop iterating over blocks of `x`, `left_proj`, `right_proj`, `left_gate`, `right_gate`, `out_gate`; and then in the epilogue, does masking, computes sigmoids, and performs elementwise multiplications. In my implementation it's called `two_mm` (because initially it only did two matmuls before I fused more computations into the kernel, and I never renamed it).

I mostly prompted Claude Code for help writing this kernel, and I was pretty impressed—it probably made fewer mistakes than I would have. I initially gave it the [09-persistent-matmul tutorial](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html) and asked it to modify it.

I don't think this kernel is very well optimized; from my memory, I vaguely recall that it was getting around 300-400 TFlops on H100 for the B=2, SL=512, dim=384 case that I was doing most of my local testing on. But at least on H100, it appeared to provide modest speedups and was better than the unfused implementation.

Here's a simplified sketch of the kernel structure:

```python
@triton.jit
def two_mm_fused_kernel(
    x_ptr, weight1_ptr, weight2_ptr, gate1_ptr, gate2_ptr, gate_out_ptr,
    out1_ptr, out2_ptr, out_gate_ptr, mask_ptr,
    M, N, K,  # M = batch*seq_len*seq_len, N = hidden_dim, K = input_dim
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    """
    Fused kernel computing:
    1. Five matmuls: x @ weight1, x @ weight2, x @ gate1, x @ gate2, x @ gate_out
    2. Apply mask to first two outputs (left_proj, right_proj)
    3. Apply sigmoid to gates
    4. Elementwise multiply: left *= left_gate, right *= right_gate
    5. Store results with permuted layout for later BMM
    """
    
    # Calculate which tile this thread block handles
    pid_m = tl.program_id(0) // num_blocks_n
    pid_n = tl.program_id(0) % num_blocks_n
    
    # Initialize accumulators for all five matmuls
    acc_left = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_right = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_left_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_right_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_out_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over K dimension, computing all five matmuls in parallel
    for k_block in range(0, K, BLOCK_K):
        # Load input block and all weight blocks
        x_block = tl.load(x_ptr + offsets_x)
        w1 = tl.load(weight1_ptr + offsets_w)
        w2 = tl.load(weight2_ptr + offsets_w)
        g1 = tl.load(gate1_ptr + offsets_w)
        g2 = tl.load(gate2_ptr + offsets_w)
        g_out = tl.load(gate_out_ptr + offsets_w)
        
        # Accumulate all matmuls using tensor cores
        acc_left = tl.dot(x_block, w1.T, acc_left, allow_tf32=True)
        acc_right = tl.dot(x_block, w2.T, acc_right, allow_tf32=True)
        acc_left_gate = tl.dot(x_block, g1.T, acc_left_gate, allow_tf32=True)
        acc_right_gate = tl.dot(x_block, g2.T, acc_right_gate, allow_tf32=True)
        acc_out_gate = tl.dot(x_block, g_out.T, acc_out_gate, allow_tf32=True)
    
    # --- Epilogue: fused operations ---
    
    # Load mask and apply to left/right projections
    mask = tl.load(mask_ptr + offsets_m)
    acc_left = tl.where(mask[:, None], acc_left, 0)
    acc_right = tl.where(mask[:, None], acc_right, 0)
    
    # Apply sigmoid activation to gates
    left_gate = sigmoid(acc_left_gate)
    right_gate = sigmoid(acc_right_gate)
    out_gate = sigmoid(acc_out_gate)
    
    # Elementwise multiply with gates
    left_final = acc_left * left_gate
    right_final = acc_right * right_gate
    
    # Store results
    tl.store(out1_ptr + out_offsets, left_final)
    tl.store(out2_ptr + out_offsets, right_final)
    tl.store(out_gate_ptr + out_offsets, out_gate)
```

### 2. Fuse/Improve the LayerNorms

**Performance after custom LayerNorms (and fixing timeout issues from step 3):** 4.58ms (47% latency reduction from baseline, 27% from step 1)

At this point I collected a kineto trace, and observed that the layernorm kernels were actually quite slow, at perhaps half of peak memory bandwidth on the H100 that I had rented.

I started with this part:

```python
out = self.to_out_norm(out)
out = out * out_gate
```

I started with torch.compile-ing this, but found that max-autotune caused timeouts and default heuristics were not much better than eager mode. (I believe the PyTorch team is working on improving the heuristics, but those improvements certainly aren't available in PyTorch 2.7.1, used for this competition)

I wrote a simple layernorm that reads a 2d block and reduces over the reduction dimension. My experience with normalizations is that as long as you have enough bytes in flight, you can generally get relatively good performance from Triton.

However, this layernorm is a special case—it is reducing over a tensor of shape `[B, SL, SL, dim]` with strides `[SL*SL*dim, SL, 1, SL*SL]` and dim=128—i.e. the reduction dimension is not contiguous. Through autotuning, Triton selected a 16×128 block with num_warps=1. I was pretty surprised by this, as in this case 16×fp32 is only 64 bytes (or 16×fp16, as I tried later, is only 32 bytes)—less than the 128-byte cache lines. I probably won't have time, but I think it would be interesting to follow up on why this is the case for Triton.

Next I substituted a Triton implementation for the first layernorm, where improved bandwidth can have an even larger overall effect on latency due to the (potentially) larger dimension which can be more than the hidden dimension of 128.

```python
x = self.norm(x)
```

Locally, I saw huge wins from this. But when I submitted to the leaderboard, I saw timeouts. I believe my first H100 submission went through, but after that I wasn't able to get any submissions to pass.

### 3. Debugging Timeouts

I spent a while submitting to the leaderboard to try to identify what I was doing wrong. I was skeptical from the beginning, because I was surprised that my only successful submission with a custom-first-layernorm took nearly twice the time as the pytorch-layernorm submission, while I expected the compile time / autotuning time of the layernorm to be relatively small. 

A few things I tried:

- **Hacking together a Triton caching server** running on a DigitalOcean site, which my submission would try to write and read from. (This didn't work)
- **Pre-computing configuration heuristics** from autotuning runs on rented H100s/B200s (This also didn't work)
- **Removing custom kernels** and replacing them with torch kernels. It appeared that the first triton kernel had the biggest effect: replacing it with PyTorch kernels seemed to improve the chance of my submissions succeeding, but not always.

With some pointers from Mark Saroufim, I tried running the [eval script](https://github.com/gpu-mode/reference-kernels/blob/main/problems/bioml/trimul/eval.py) with the benchmark data and collecting [profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) traces.

I found two things:

1. The [Cauchy distribution](https://github.com/gpu-mode/reference-kernels/blob/main/problems/bioml/trimul/reference.py#L137-L139) is computed on CPU (and then transferred to GPU) and it takes a *long* time—on [2, 128, 128, 256], it took ~110ms. And it gets called **on each repetition of the kernel during profiling**.
2. The profiling loop generally runs 100 times, but [**can exit early if the custom kernel experiences low variance**](https://github.com/gpu-mode/reference-kernels/blob/main/problems/bioml/trimul/eval.py#L252-L255)

My hypothesis is that the additional CPU overhead for launching a Triton kernel (vs. a standard PyTorch kernel) made my implementation have, on average, higher latency variation, triggering the full 100 iterations during benchmarking on a few of the Cauchy distribution cases.

I submitted a [PR to move the Cauchy distribution calculation onto GPU](https://github.com/gpu-mode/reference-kernels/pull/66), which the GPU Mode team graciously considered, but they ultimately decided to instead just increase the timeout limit (I imagine it is not fair to adjust the eval script in the middle of the contest!)

### 4. Einsum to BMM

**Performance after einsum to BMM conversion:** 2.73ms (68% latency reduction from baseline, 40% from step 2)

In a profiler trace, I observed that the einsum introduced a bunch of data layout permutations (on GPU), and the gemm kernel itself was somewhat slow.

The einsum:

```python
out = einsum('... i k d, ... j k d -> ... i j d', left, right)
```

is really just a bmm but with some transpositions required. I decided to switch this to a bmm.

To start off, I introduced some data transformations in torch to get the layout I wanted. In particular, I wanted:

```python
left = left.permute(0, 3, 1, 2)  # [b d i k] or [b d sl1 sl2], stride (contiguous)
right = right.permute(0, 3, 1, 2)  # similar transformation
```

I added these transformations and then replaced the einsum with a bmm.

Next, I **fused the permutations into the preceding `two_mm` kernel**, which ended up being mainly a bunch of indexing logic for computing the offsets of the outputs. (Claude didn't get the right implementation, but I implemented a buggy version and Claude fixed it)

The key part of the fused permutation logic:

```python
# Decompose flattened indices back to 4D coordinates
off_c_batch = offs_cm // (seq_len * seq_len)
off_c_sl1 = (offs_cm // seq_len) % seq_len
off_c_sl2 = offs_cm % seq_len
off_c_dim = offs_cn

# Calculate offsets for transposed layout [B, D, SL, SL] instead of [B, SL, SL, D]
c_offsets = (
    (off_c_batch * stride_c0 + off_c_sl1 * stride_c1 + off_c_sl2 * stride_c2)[:, None] + 
    off_c_dim[None, :] * stride_c3
)
```

### 5. Data Types

**Performance fp16 usage:** 1.91ms (77% latency reduction from baseline, 30% from step 2)

The problem has relatively lenient accuracy requirements (atol, rtol of 2e-2). So I converted everything except the input and the output to fp16. This has two benefits: faster matmuls and reduced memory transfer.

I don't remember how much speedup we get from this, but it was a lot. The previous custom kernels/fusions were helpful to allow us to fuse all the dtype conversions into kernels and actually benefit from the reduced memory transfers.

### 6. Failed Attempt at A100 / MI300

With a few hours left in the contest I revisited the problem with the goal of submitting some attempts at A100 or MI300.

I couldn't get MI300 with autotuning to pass the leaderboard tests without timing out. I also wasn't able to find a cheap MI300 to do local autotuning on (and I'm not sure whether this was an issue with the implementation taking a long time or a problem with long-running Cauchy data generation), so instead I just aimlessly submitted various alternative implementations hoping that one would get lucky. Unfortunately, all of my submissions still timed out.

I was able to get one A100 submission to pass the timeout issues—it wasn't the fastest on the leaderboard however.

With a vast.ai rental, I observed with torch.profile that `two_mm` was the obvious bottleneck.

![A100 profiler trace showing two_mm as the bottleneck](https://davidberard98.github.io/gpumode-trimul/long-a100-twomm.png)

I then wasted an hour trying to find a cloud provider with A100s that supported NCU.

## Other Ideas

If I had a bit more time, I would want to do a few more investigations:

- **H100/B200 autotuning on the fused matmul kernel**: I submitted two versions for each of H100 and B200: one in which the fused matmul kernel is autotuned at runtime, and one where I ran some benchmarking ahead of time and generated heuristics for selecting configs for the fused matmul kernel. The autotuned version was faster in both cases—from 1158µs to 1088µs on Blackwell, and from 1533µs to 1371µs on Hopper. I'm wondering where the difference comes from—bad benchmarking for my heuristic selection? Power throttling? Different hardware?

- **Better fused matmul kernel**: I chose an implementation arbitrarily, and I expect there's a lot of room for improvement. I think an NCU trace would be super helpful to understand where the performance bottlenecks are.

- **Fusions with norms**: [Arseni Ivanov](https://arseniivanov.github.io/blog.html)'s implementation (fusing the layernorm into the matmuls) is pretty interesting and I'd be interested in exploring this approach further.

## Conclusion

This was a super fun problem! Overall, my takeaways were:
* Start with a torch.profile to understand what's taking a long time. I didn't realize at first that the layernorms were so slow, but this was one of the easiest and most impactful changes I made.
* Claude does a pretty good job with Triton kernels
* Cloud providers supporting NCU are hard to find
* Benchmarking is hard!

---

*Challenge details: [https://tinyurl.com/gpumode-trimul](https://tinyurl.com/gpumode-trimul)*
