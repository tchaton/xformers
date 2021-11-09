# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import torch
import triton
import triton.language as tl
import math


# autotune: Triton will test out these configurations, and automatically pick the fastest one.
# heuristic: add arguments to the kernel call automatically given some heuristics. These arguments are passed in "meta"
# fmt: off
@triton.heuristics(values={"cols": lambda *args, **_: triton.next_power_of_2(args[-1])})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=["K"],
)
@triton.jit
def _bandwidth(
    Y, X,
    stride_ym,
    stride_xm,
    K,
    **META,  # extra parameters which can be automatically filled in given some heuristics
):
    # fmt: om

    """
    No-op kernel, test the max bandwidth that we can reach.
    The input buffer is expected to be M x K
    We start one thread block per row
    """

    m = tl.program_id(0)

    # col indices
    k = tl.arange(0, META["cols"])

    # the memory address of all the elements that we want to load can be computed as follows
    x_ptrs = X + m * stride_xm + k

    # Mask half of the reads, arbitrary
    io_mask = True  # No-op
    if META["mask"]:
        K_half = K // 2
        io_mask = io_mask & (k <= K_half)

    if META["read"]:
        x = tl.load(x_ptrs, mask=io_mask, other=float("-inf"))
    else:
        x = 0.

    # Do something with x
    x *= 0.

    if META["write"]:
        # write back to Y.
        # we only write once, hence the "fused" softmax naming
        y_ptrs = Y + m * stride_ym + k

        # technically we could write only the lower triangular matrix in the causal case
        # but this is deemed to error prone
        tl.store(y_ptrs, x, mask=io_mask)


def bandwidth(x: torch.Tensor, y: torch.Tensor, read: bool, write: bool, mask: bool):
    # The buffer needs to be a power of two (Triton requirement)
    # Check that the input value makes sense
    rows, cols = x.shape

    assert x.shape[0] == y.shape[0] and x.shape[1] == y.shape[1]
    assert math.log2(cols).is_integer(), \
        "This test only works with powers of two columns"

    # Kick the kernels
    grid = (rows,)  # just 1D, one thread block per row

    _bandwidth[grid](
        y, x,   # pointers to the buffers
        y.stride(0), x.stride(0),  # the strides
        cols,  # K above
        read=read,
        write=write,
        mask=mask
        )

    return True
