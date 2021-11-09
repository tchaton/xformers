# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, Dict

import torch
import triton

from xformers.benchmarks.utils import pretty_plot, pretty_print
from xformers.triton.k_bandwidth import bandwidth

SHAPES = [
    (384, 128),
    (784, 512),
    (1024, 1024),
    (1024, 2048),
    (1024, 4096),
    (1024, 8192),
]


def get_triton_callback(read: bool, write: bool, mask: bool):
    def callback(x, y):
        _ = bandwidth(x, y, read=read, write=write, mask=mask)

    return callback


def to_gbs(a, ms, read: bool, write: bool):
    "Get gigaBit/s given a time and buffer size"
    multiple = 2 if read and write else 1

    # ms -> s
    # giga
    return (multiple * a.numel() * a.element_size() * 1e-9) / (ms * 1e-3)


def do_bench(shapes=SHAPES):
    results: Dict[str, Any] = {}

    for read in [True, False]:
        for write in [True, False]:
            if not read and not write:
                continue

            for masked in [True, False]:
                for shape in shapes:
                    a = torch.rand(
                        shape,
                        device=torch.device("cuda"),
                        dtype=torch.float32,
                        requires_grad=False,
                    )

                    b = a.clone()

                    callback = get_triton_callback(read, write, masked)

                    time = triton.testing.do_bench(lambda: callback(a, b))[0]

                    metric = to_gbs(a, time, read, write)

                    shape_key = "".join(f"{s}x" for s in shape)[:-1]
                    if shape_key not in results:
                        results[shape_key] = {}

                    testcase_name = f"Read:{read} - Write: {write} - Masked {masked}"

                    results[shape_key][testcase_name] = f"{metric:.1f}"

    title = "Bandwidth measurements"
    units = "GB/s"
    pretty_print(
        results,
        title=title,
        units=units,
    )

    pretty_plot(results, title, units, dash_key="Masked True")


if __name__ == "__main__":
    do_bench()
