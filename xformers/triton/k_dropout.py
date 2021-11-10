# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


# CREDITS: This comes almost as-is from the Triton dropout tutorial
# https://raw.githubusercontent.com/openai/triton/master/python/tutorials/04-low-memory-dropout.py

import triton
import triton.language as tl

_k_configs = [
    triton.Config({"BLOCK_SIZE": 256}, num_stages=3, num_warps=1),
    triton.Config({"BLOCK_SIZE": 512}, num_stages=3, num_warps=2),
    triton.Config({"BLOCK_SIZE": 1024}, num_stages=3, num_warps=4),
    triton.Config({"BLOCK_SIZE": 2048}, num_stages=3, num_warps=8),
    triton.Config({"BLOCK_SIZE": 4096}, num_stages=3, num_warps=16),
]


@triton.jit
def _dropmask(SEEDS, row, p, offsets):
    # randomly prune the weights
    seed = SEEDS + row
    random = tl.rand(seed.to(tl.int32), offsets)
    x_keep = random > p

    return x_keep


# fmt: off
@triton.autotune(
    configs=_k_configs,
    key=["N"],
)
@triton.jit
def k_dropout_fw(
    Y, MASK, X, BIAS, SEEDS,
    stride,
    N,
    p,
    **META,
):
    """
    Apply dropout on an input tensor
    Y : Output (M, N)
    X : Input (M, N)
    S : Seeds (M,)
    p : dropout probability
    """
    # fmt: on

    BLOCK_SIZE = META["BLOCK_SIZE"]
    row = tl.program_id(axis=0)
    col = tl.program_id(axis=1)

    # compute memory offsets of elements handled by this instance
    offsets = row * stride + col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) < N

    # load data from x
    x_ptrs = X + offsets
    x = tl.load(x_ptrs, mask=mask)

    # optionally apply a fused bias
    if META["USE_BIAS"]:
        b_ptrs = BIAS + col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        b = tl.load(b_ptrs, mask=mask)
        x += b

    # optional: fused activation (while the data is in shared memory)
    if META["ACTIVATION"]:
        x = META["ACTIVATION"](x)

    # randomly prune it
    if p > 0.:
        x_keep = _dropmask(SEEDS, row, p, offsets)
        zero = 0.0
        output = tl.where(x_keep, (x / (1 - p)).to(x.dtype),  zero.to(x.dtype))

        if META["SAVE_MASK"]:
            mask_ptrs = MASK + offsets
            tl.store(mask_ptrs, x_keep , mask=mask)

    else:
        output = x

    y_ptrs = Y + offsets
    tl.store(y_ptrs, output, mask=mask)


# fmt: off
@triton.autotune(
    configs=_k_configs,
    key=["N"],
)
@triton.jit
def k_dropout_bw(
    GRAD_IN, GRAD_OUT, INPUTS, MASK, BIAS, SEEDS,
    stride_grad, stride_inputs,
    N,
    p,
    **META,
):
    """
    Apply dropout on an input tensor
    GRAD_OUT    (M, N)
    GRAD_IN     (M, N)
    BIAS        (N,)
    SEEDS       (M,)
    p : dropout probability
    """
    # fmt: on

    BLOCK_SIZE = META["BLOCK_SIZE"]
    row = tl.program_id(axis=0)
    col = tl.program_id(axis=1)

    # compute memory offsets of elements handled by this instance
    grad_offsets = row * stride_grad + col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) < N

    # load the incoming grad data
    grad_out_ptrs = GRAD_OUT + grad_offsets
    grad_out = tl.load(grad_out_ptrs, mask=mask)

    # optional: fused activation (while the data is in shared memory)
    if META["ACTIVATION_GRAD"]:
        # Recompute the activation inputs, more memory efficient
        input_ptrs = INPUTS + row * stride_inputs + col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        inputs = tl.load(input_ptrs, mask=mask)

        # optionally apply a fused bias
        if META["USE_BIAS"]:
            b_ptrs = BIAS + col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            b = tl.load(b_ptrs, mask=mask)
            inputs += b

        act_grad = META["ACTIVATION_GRAD"](inputs)
        grad_out *= act_grad

    # randomly prune it
    if p > 0.:
        if META["SAVED_MASK"]:
            # Reload the activation inputs, faster
            mask_ptrs = MASK + row * stride_inputs + col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            dropmask = tl.load(mask_ptrs, mask=mask)
        else:
            dropmask = _dropmask(SEEDS, row, p, grad_offsets)

        zero = 0.0
        output = tl.where(dropmask, (grad_out / (1 - p)).to(grad_out.dtype), zero.to(grad_out.dtype))
    else:
        output = grad_out

    # write-back
    y_ptrs = GRAD_IN + grad_offsets
    tl.store(y_ptrs, output, mask=mask)
