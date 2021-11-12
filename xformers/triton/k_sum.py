# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 2}, num_warps=1),
        triton.Config({"BLOCK_N": 2}, num_warps=2),
        triton.Config({"BLOCK_N": 2}, num_warps=4),
        triton.Config({"BLOCK_N": 2}, num_warps=8),
        triton.Config({"BLOCK_N": 2}, num_warps=16),
        triton.Config({"BLOCK_N": 2}, num_warps=32),
    ],
    key=["M", "N"],
)
@triton.heuristics(
    values={"BLOCK_SIZE": lambda *args, **_: triton.next_power_of_2(args[-1])}
)
@triton.jit
def k_sum_0(
    Y,
    X,
    stride_xm,
    M,
    N,
    **meta,  # extra parameters which can be automatically filled in given some heuristics
):
    # fmt: om

    """
    Sum a 2d tensor over the first dimension
    """

    # The columns are independent, so we parallelize across them
    n = tl.program_id(0)

    # row indices
    m = tl.arange(0, meta["BLOCK_SIZE"])

    # To get some extra parallelization, we can try to handle several columns in the same thread block
    rn = n * meta["BLOCK_N"] + tl.arange(0, meta["BLOCK_N"])

    # the memory address of all the elements that we want to load can be computed as follows
    x_ptrs = X + m[:, None] * stride_xm + rn[None, :]

    # load input data; pad out-of-bounds elements with 0
    x = tl.load(x_ptrs, mask=(m[:, None] < M) & (rn[None, :] < N), other=0.0)

    x_sum = tl.sum(x, 0)
    tl.store(Y + rn, x_sum, mask=rn < N)
