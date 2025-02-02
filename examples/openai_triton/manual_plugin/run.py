import argparse
import json
import math
from pathlib import Path

import torch
from fmha_triton import fused_attention
from plugin import get_engine_name

from tensorrt_llm import profiler
from tensorrt_llm._utils import (str_dtype_to_torch, str_dtype_to_trt,
                                 trt_dtype_to_torch)
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime.session import Session


def run(engine_dir,
        batch_size,
        seq_len,
        num_heads,
        head_size,
        do_benchmark=False):
    # Load trt engine.
    engine_dir = Path(engine_dir)
    config_path = engine_dir / 'config.json'
    with config_path.open('r') as f:
        config = json.load(f)
    dtype = config['builder_config']['precision']
    serialize_path = engine_dir / get_engine_name(head_size, dtype)

    with open(serialize_path, 'rb') as f:
        session = Session.from_serialized_engine(f.read())

    # Prepare input tensors.
    torch_dtype = str_dtype_to_torch(dtype) if isinstance(dtype, str) else dtype
    shape = (batch_size, num_heads, seq_len, head_size)
    q = torch.normal(mean=0.1,
                     std=0.2,
                     size=shape,
                     dtype=torch_dtype,
                     device='cuda')
    k = torch.normal(mean=0.4,
                     std=0.2,
                     size=shape,
                     dtype=torch_dtype,
                     device='cuda')
    v = torch.normal(mean=0.3,
                     std=0.2,
                     size=shape,
                     dtype=torch_dtype,
                     device='cuda')
    inputs = {'Q': q, 'K': k, 'V': v}

    # Prepare output tensors.
    output_info = session.infer_shapes([
        TensorInfo(name, str_dtype_to_trt(dtype), tensor.shape)
        for name, tensor in inputs.items()
    ])
    logger.debug(f'output info {output_info}')
    outputs = {
        t.name: torch.empty(tuple(t.shape),
                            dtype=trt_dtype_to_torch(t.dtype),
                            device='cuda')
        for t in output_info
    }

    # Execute model inference
    stream = torch.cuda.current_stream()
    ok = session.run(inputs=inputs, outputs=outputs, stream=stream.cuda_stream)
    assert ok, 'Engine execution failed'

    # Sanity check
    stream.synchronize()
    sm_scale = 1.0 / math.sqrt(head_size)
    ref = fused_attention(q, k, v, sm_scale)
    out = outputs["out"]
    logger.debug(
        f'Out: vals: {out.view(1, -1)} abs_sum: {out.float().abs().sum()}')
    logger.debug(
        f'Ref: vals: {ref.view(1, -1)} abs_sum: {ref.float().abs().sum()}')
    torch.testing.assert_close(out, ref)

    if do_benchmark:
        n_repeats = 10

        # For fair comparison, pre-allocate buffers as trt plugin does.
        shape = (q.shape[0] * q.shape[1], q.shape[2])
        L = torch.empty(shape, device=q.device, dtype=torch.float32)
        m = torch.empty(shape, device=q.device, dtype=torch.float32)
        o = torch.empty_like(q)

        # Triton warm-up
        fused_attention(q, k, v, sm_scale, l_buf=L, m_buf=m, o_buf=o)
        stream.synchronize()
        for _ in range(n_repeats):
            profiler.start('Triton')
            fused_attention(q, k, v, sm_scale, l_buf=L, m_buf=m, o_buf=o)
            stream.synchronize()
            profiler.stop('Triton')

        # TRT warm-up
        stream.synchronize()
        ok = session.run(inputs=inputs,
                         outputs=outputs,
                         stream=stream.cuda_stream)
        stream.synchronize()
        for _ in range(n_repeats):
            profiler.start('TRT Plugin')
            ok = session.run(inputs=inputs,
                             outputs=outputs,
                             stream=stream.cuda_stream)
            stream.synchronize()
            profiler.stop('TRT Plugin')
            assert ok
        profiler.summary()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--engine_dir', type=str, required=True,
                        help='Path to engine directory')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--seq_len', type=int, default=128,
                        help='Sequence length')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number
