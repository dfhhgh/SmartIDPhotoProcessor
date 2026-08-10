"""
ONNX weight extraction and mapping to PyTorch BiSeNet state_dict.

The ONNX model has BatchNorm fused into Conv layers. During ONNX export,
Conv+BN was folded so that:
  fused_weight = original_weight * gamma / sqrt(running_var + eps)
  fused_bias = (original_bias - running_mean) * gamma / sqrt(running_var + eps) + beta

The PyTorch model uses Conv(bias=False) + BN layers (upstream architecture).
To correctly reproduce ONNX output, we load:
  - fused_weight -> Conv(bias=False).weight
  - fused_bias -> BN.bias (with BN.gamma=1, running_mean=0, running_var=1)
This makes BN act as: output = 1*(x - 0)/1 + fused_bias = x + fused_bias
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import onnx
import torch

logger = logging.getLogger(__name__)

# ONNX Conv node name -> PyTorch Conv weight prefix.
_ONNX_TO_PYTORCH_CONV: dict[str, str] = {
    "/fpn/backbone/conv1/Conv": "cp.resnet.conv1",
    "/fpn/backbone/layer1/layer1.0/conv1/Conv": "cp.resnet.layer1.0.conv1",
    "/fpn/backbone/layer1/layer1.0/conv2/Conv": "cp.resnet.layer1.0.conv2",
    "/fpn/backbone/layer1/layer1.1/conv1/Conv": "cp.resnet.layer1.1.conv1",
    "/fpn/backbone/layer1/layer1.1/conv2/Conv": "cp.resnet.layer1.1.conv2",
    "/fpn/backbone/layer2/layer2.0/conv1/Conv": "cp.resnet.layer2.0.conv1",
    "/fpn/backbone/layer2/layer2.0/conv2/Conv": "cp.resnet.layer2.0.conv2",
    "/fpn/backbone/layer2/layer2.0/downsample/downsample.0/Conv": "cp.resnet.layer2.0.downsample.0",
    "/fpn/backbone/layer2/layer2.1/conv1/Conv": "cp.resnet.layer2.1.conv1",
    "/fpn/backbone/layer2/layer2.1/conv2/Conv": "cp.resnet.layer2.1.conv2",
    "/fpn/backbone/layer3/layer3.0/conv1/Conv": "cp.resnet.layer3.0.conv1",
    "/fpn/backbone/layer3/layer3.0/conv2/Conv": "cp.resnet.layer3.0.conv2",
    "/fpn/backbone/layer3/layer3.0/downsample/downsample.0/Conv": "cp.resnet.layer3.0.downsample.0",
    "/fpn/backbone/layer3/layer3.1/conv1/Conv": "cp.resnet.layer3.1.conv1",
    "/fpn/backbone/layer3/layer3.1/conv2/Conv": "cp.resnet.layer3.1.conv2",
    "/fpn/backbone/layer4/layer4.0/conv1/Conv": "cp.resnet.layer4.0.conv1",
    "/fpn/backbone/layer4/layer4.0/conv2/Conv": "cp.resnet.layer4.0.conv2",
    "/fpn/backbone/layer4/layer4.0/downsample/downsample.0/Conv": "cp.resnet.layer4.0.downsample.0",
    "/fpn/backbone/layer4/layer4.1/conv1/Conv": "cp.resnet.layer4.1.conv1",
    "/fpn/backbone/layer4/layer4.1/conv2/Conv": "cp.resnet.layer4.1.conv2",
    "/fpn/arm32/conv_block/conv/Conv": "cp.arm32.conv_block.conv",
    "/fpn/arm32/attention/attention.0/Conv": "cp.arm32.conv_atten",
    "/fpn/arm16/conv_block/conv/Conv": "cp.arm16.conv_block.conv",
    "/fpn/arm16/attention/attention.0/Conv": "cp.arm16.conv_atten",
    "/fpn/conv_avg/conv/Conv": "cp.conv_avg.conv",
    "/fpn/conv_head32/conv/Conv": "cp.conv_head32.conv",
    "/fpn/conv_head16/conv/Conv": "cp.conv_head16.conv",
    "/ffm/conv_block/conv/Conv": "ffm.convblk.conv",
    "/ffm/conv1/Conv": "ffm.conv1",
    "/ffm/conv2/Conv": "ffm.conv2",
    "/conv_out/conv_block/conv/Conv": "conv_out.convblk.conv",
    "/conv_out/conv/Conv": "conv_out.conv",
    "/conv_out16/conv_block/conv/Conv": "conv_out16.convblk.conv",
    "/conv_out16/conv/Conv": "conv_out16.conv",
    "/conv_out32/conv_block/conv/Conv": "conv_out32.convblk.conv",
    "/conv_out32/conv/Conv": "conv_out32.conv",
}

# ONNX Conv node name -> corresponding BN parameter prefix.
# This mapping accounts for the fact that BasicBlock uses bn1/bn2 naming,
# while ConvBNReLU modules use just "bn".
_ONNX_TO_PYTORCH_BN: dict[str, str] = {
    "/fpn/backbone/conv1/Conv": "cp.resnet.bn1",
    "/fpn/backbone/layer1/layer1.0/conv1/Conv": "cp.resnet.layer1.0.bn1",
    "/fpn/backbone/layer1/layer1.0/conv2/Conv": "cp.resnet.layer1.0.bn2",
    "/fpn/backbone/layer1/layer1.1/conv1/Conv": "cp.resnet.layer1.1.bn1",
    "/fpn/backbone/layer1/layer1.1/conv2/Conv": "cp.resnet.layer1.1.bn2",
    "/fpn/backbone/layer2/layer2.0/conv1/Conv": "cp.resnet.layer2.0.bn1",
    "/fpn/backbone/layer2/layer2.0/conv2/Conv": "cp.resnet.layer2.0.bn2",
    "/fpn/backbone/layer2/layer2.0/downsample/downsample.0/Conv": "cp.resnet.layer2.0.downsample.1",
    "/fpn/backbone/layer2/layer2.1/conv1/Conv": "cp.resnet.layer2.1.bn1",
    "/fpn/backbone/layer2/layer2.1/conv2/Conv": "cp.resnet.layer2.1.bn2",
    "/fpn/backbone/layer3/layer3.0/conv1/Conv": "cp.resnet.layer3.0.bn1",
    "/fpn/backbone/layer3/layer3.0/conv2/Conv": "cp.resnet.layer3.0.bn2",
    "/fpn/backbone/layer3/layer3.0/downsample/downsample.0/Conv": "cp.resnet.layer3.0.downsample.1",
    "/fpn/backbone/layer3/layer3.1/conv1/Conv": "cp.resnet.layer3.1.bn1",
    "/fpn/backbone/layer3/layer3.1/conv2/Conv": "cp.resnet.layer3.1.bn2",
    "/fpn/backbone/layer4/layer4.0/conv1/Conv": "cp.resnet.layer4.0.bn1",
    "/fpn/backbone/layer4/layer4.0/conv2/Conv": "cp.resnet.layer4.0.bn2",
    "/fpn/backbone/layer4/layer4.0/downsample/downsample.0/Conv": "cp.resnet.layer4.0.downsample.1",
    "/fpn/backbone/layer4/layer4.1/conv1/Conv": "cp.resnet.layer4.1.bn1",
    "/fpn/backbone/layer4/layer4.1/conv2/Conv": "cp.resnet.layer4.1.bn2",
    "/fpn/arm32/conv_block/conv/Conv": "cp.arm32.conv_block.bn",
    "/fpn/arm32/attention/attention.0/Conv": "cp.arm32.bn_atten",
    "/fpn/arm16/conv_block/conv/Conv": "cp.arm16.conv_block.bn",
    "/fpn/arm16/attention/attention.0/Conv": "cp.arm16.bn_atten",
    "/fpn/conv_avg/conv/Conv": "cp.conv_avg.bn",
    "/fpn/conv_head32/conv/Conv": "cp.conv_head32.bn",
    "/fpn/conv_head16/conv/Conv": "cp.conv_head16.bn",
    "/ffm/conv_block/conv/Conv": "ffm.convblk.bn",
    "/conv_out/conv_block/conv/Conv": "conv_out.convblk.bn",
    "/conv_out16/conv_block/conv/Conv": "conv_out16.convblk.bn",
    "/conv_out32/conv_block/conv/Conv": "conv_out32.convblk.bn",
}

# The 5 named Conv nodes have NO BN in the upstream code (bias=False, no BN).
_NO_BN_NODES: set[str] = {
    "/ffm/conv1/Conv",
    "/ffm/conv2/Conv",
    "/conv_out/conv/Conv",
    "/conv_out16/conv/Conv",
    "/conv_out32/conv/Conv",
}


def _numpy_from_initializer(init: onnx.TensorProto) -> np.ndarray:
    return onnx.numpy_helper.to_array(init)


def load_onnx_to_pytorch(
    onnx_path: Path,
    pytorch_model: torch.nn.Module,
) -> torch.nn.Module:
    """Load ONNX weights into PyTorch BiSeNet, handling BN fusion correctly."""
    model_proto = onnx.load(str(onnx_path))
    graph = model_proto.graph
    init_index = {init.name: init for init in graph.initializer}

    state_dict = pytorch_model.state_dict()
    loaded_keys: set[str] = set()

    for onnx_node_name, conv_prefix in _ONNX_TO_PYTORCH_CONV.items():
        conv_node = None
        for node in graph.node:
            if node.name == onnx_node_name and node.op_type == "Conv":
                conv_node = node
                break

        if conv_node is None:
            logger.warning("ONNX Conv node not found: %s", onnx_node_name)
            continue

        inputs = list(conv_node.input)
        weight = _numpy_from_initializer(init_index[inputs[1]])

        weight_key = f"{conv_prefix}.weight"
        if weight_key in state_dict:
            state_dict[weight_key] = torch.from_numpy(weight.copy())
            loaded_keys.add(weight_key)

        has_fused_bias = len(inputs) >= 3
        is_no_bn = onnx_node_name in _NO_BN_NODES

        if has_fused_bias and not is_no_bn:
            fused_bias = _numpy_from_initializer(init_index[inputs[2]])
            bn_prefix = _ONNX_TO_PYTORCH_BN[onnx_node_name]

            for suffix, fill in [
                (".bias", fused_bias),
                (".weight", np.ones_like(fused_bias)),
                (".running_mean", np.zeros_like(fused_bias)),
                (".running_var", np.ones_like(fused_bias)),
            ]:
                key = f"{bn_prefix}{suffix}"
                if key in state_dict:
                    if isinstance(fill, np.ndarray):
                        state_dict[key] = torch.from_numpy(fill.copy())
                    else:
                        state_dict[key] = torch.full_like(state_dict[key], fill)
                    loaded_keys.add(key)

            nbt_key = f"{bn_prefix}.num_batches_tracked"
            if nbt_key in state_dict:
                state_dict[nbt_key] = torch.tensor(0, dtype=torch.long)
                loaded_keys.add(nbt_key)

    # Load named initializers (ffm.conv1.weight, conv_out.conv.weight, etc.)
    for init in graph.initializer:
        for key in state_dict:
            if key.endswith(init.name) and key not in loaded_keys:
                arr = _numpy_from_initializer(init)
                state_dict[key] = torch.from_numpy(arr.copy())
                loaded_keys.add(key)

    missing, unexpected = pytorch_model.load_state_dict(state_dict, strict=False)

    logger.info(
        "Weight loading: %d/%d keys loaded, %d missing, %d unexpected",
        len(loaded_keys),
        len(state_dict),
        len(missing),
        len(unexpected),
    )

    if missing:
        logger.warning("Missing keys: %s", missing[:30])
    if unexpected:
        logger.warning("Unexpected keys: %s", unexpected[:30])

    pytorch_model.eval()
    return pytorch_model
