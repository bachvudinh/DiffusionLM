"""Model weight loading utilities.

Derived from JetEngine by Yihan Bian et al.
Reference: https://github.com/Labman42/JetEngine

Architecture Overview
====================

    Handles loading model weights from safetensors files and HuggingFace
    models, with support for tensor parallel sharding and fused MoE weights.

    ┌────────────────────────────────────────────────────────────────────┐
    │                     Weight Loading Pipeline                        │
    │                                                                    │
    │  load_model(model, path)                                           │
    │         │                                                          │
    │         ├── _prepare_fused_tensors()                                │
    │         │     └── Pre-allocate _w1, _w2 for MoE expert weights     │
    │         │                                                          │
    │         ├── For each *.safetensors file:                            │
    │         │     ├── Check packed_modules_mapping                      │
    │         │     │     (q_proj→qkv_proj, gate_proj→gate_up_proj)      │
    │         │     ├── MoE expert? → _load_expert_weight_to_fused()     │
    │         │     └── Regular? → param.weight_loader(param, tensor)     │
    │         │           └── Handles TP sharding automatically          │
    │         │                                                          │
    │  load_from_hf_model(target, hf_model)                              │
    │         └── Direct weight transfer from HF model in memory         │
    └────────────────────────────────────────────────────────────────────┘

    Input:
        model: nn.Module — target model with weight_loader methods
        path:  str       — directory containing *.safetensors files

    Output:
        Model with loaded and sharded weights (in-place)
"""

import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open
import torch.distributed as dist


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def _register_empty_parameter(module, name: str):
    empty = nn.Parameter(torch.empty(0, device='meta'), requires_grad=False)
    module.register_parameter(name, empty)


def _prepare_fused_tensors(model: nn.Module, device: torch.device | str = "cuda"):
    """Pre-allocate fused weight tensors for MoE experts."""
    for module in model.modules():
        if not (hasattr(module, "experts") and module.experts):
            continue

        exp0 = module.experts[0]
        if hasattr(exp0, "gate_up_proj"):
            shape = (len(module.experts),) + exp0.gate_up_proj.weight.shape
            module.register_buffer("_w1",
                                   torch.empty(shape,
                                               dtype=exp0.gate_up_proj.weight.dtype,
                                               device=device),
                                   persistent=False)

        if hasattr(exp0, "down_proj"):
            shape = (len(module.experts),) + exp0.down_proj.weight.shape
            module.register_buffer("_w2",
                                   torch.empty(shape,
                                               dtype=exp0.down_proj.weight.dtype,
                                               device=device),
                                   persistent=False)

        for expert in module.experts:
            if hasattr(expert, "gate_up_proj"):
                _register_empty_parameter(expert.gate_up_proj, "weight")
            if hasattr(expert, "down_proj"):
                _register_empty_parameter(expert.down_proj, "weight")


def _is_moe_expert_weight(weight_name: str) -> bool:
    return 'experts.' in weight_name and ('gate_up_proj' in weight_name or 'down_proj' in weight_name)


def _load_expert_weight_to_fused(model: nn.Module, weight_name: str,
                                  weight_tensor: torch.Tensor, shard_id=None):
    """Load expert weight directly into the fused tensor with TP support."""
    parts = weight_name.split('.')
    layer_path = []
    expert_idx = None
    proj_type = None

    for i, part in enumerate(parts):
        if part == 'experts':
            expert_idx = int(parts[i + 1])
            proj_type = parts[i + 2]
            layer_path = parts[:i]
            break

    if expert_idx is None:
        return

    moe_module = model
    for attr in layer_path:
        moe_module = getattr(moe_module, attr)

    tp_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    tp_rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

    if proj_type == 'gate_up_proj' and hasattr(moe_module, '_w1'):
        fused_tensor = moe_module._w1
        local_out = fused_tensor.shape[1]
        if shard_id is None:
            if weight_tensor.shape[0] == local_out:
                fused_tensor[expert_idx].copy_(weight_tensor)
            else:
                local_weight = weight_tensor.narrow(0, tp_rank * local_out, local_out)
                fused_tensor[expert_idx].copy_(local_weight)
        else:
            half_local = local_out // 2
            if weight_tensor.shape[0] == half_local:
                start_idx = shard_id * half_local
                fused_tensor[expert_idx, start_idx:start_idx + half_local].copy_(weight_tensor)
            else:
                global_half = weight_tensor.shape[0]
                local_weight = weight_tensor.narrow(0, tp_rank * half_local, half_local)
                start_idx = shard_id * half_local
                fused_tensor[expert_idx, start_idx:start_idx + half_local].copy_(local_weight)
    elif proj_type == 'down_proj' and hasattr(moe_module, '_w2'):
        fused_tensor = moe_module._w2
        local_in = fused_tensor.shape[2]
        if weight_tensor.shape[1] == local_in:
            fused_tensor[expert_idx].copy_(weight_tensor)
        else:
            local_weight = weight_tensor.narrow(1, tp_rank * local_in, local_in)
            fused_tensor[expert_idx].copy_(local_weight)


def load_model(model: nn.Module, path: str):
    """Load model weights from safetensors files.

    Input:
        model: nn.Module — model with weight_loader methods on parameters
        path:  str       — directory containing *.safetensors files

    Output:
        None (model weights loaded in-place)
    """
    _prepare_fused_tensors(model)

    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            names = [name for name in f.keys() if 'weight' in name or 'bias' in name]
            for weight_name in names:
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        if _is_moe_expert_weight(param_name):
                            _load_expert_weight_to_fused(model, param_name, f.get_tensor(weight_name), shard_id)
                        else:
                            param = model.get_parameter(param_name)
                            weight_loader = getattr(param, "weight_loader")
                            weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    if _is_moe_expert_weight(weight_name):
                        _load_expert_weight_to_fused(model, weight_name, f.get_tensor(weight_name))
                    else:
                        param = model.get_parameter(weight_name)
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, f.get_tensor(weight_name))


def load_from_hf_model(target_model: nn.Module, hf_model: nn.Module):
    """Load weights from a HuggingFace model directly in memory.

    Input:
        target_model: nn.Module — vdllm model to load weights into
        hf_model:     nn.Module — HuggingFace model with weights

    Output:
        None (target_model weights loaded in-place)
    """
    device = next(hf_model.parameters()).device
    loaded_params = set()
    missing_params = []
    new_state_dict = {}

    for name, param in target_model.named_parameters():
        with torch.no_grad():
            if 'qkv_proj.weight' in name:
                base_name = name.replace('qkv_proj.weight', '')
                q_name = base_name + 'q_proj.weight'
                k_name = base_name + 'k_proj.weight'
                v_name = base_name + 'v_proj.weight'
                try:
                    q_weight = hf_model.get_parameter(q_name)
                    k_weight = hf_model.get_parameter(k_name)
                    v_weight = hf_model.get_parameter(v_name)
                    qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
                    target_device = device if param.is_meta else param.device
                    new_state_dict[name] = qkv_weight.to(device=target_device, dtype=param.dtype)
                    loaded_params.add(name)
                except AttributeError:
                    missing_params.append(name)
                    if not param.is_meta:
                        new_state_dict[name] = param.data

            elif 'gate_up_proj.weight' in name and 'experts.' not in name:
                base_name = name.replace('gate_up_proj.weight', '')
                gate_name = base_name + 'gate_proj.weight'
                up_name = base_name + 'up_proj.weight'
                try:
                    gate_weight = hf_model.get_parameter(gate_name)
                    up_weight = hf_model.get_parameter(up_name)
                    gate_up_weight = torch.cat([gate_weight, up_weight], dim=0)
                    target_device = device if param.is_meta else param.device
                    new_state_dict[name] = gate_up_weight.to(device=target_device, dtype=param.dtype)
                    loaded_params.add(name)
                except AttributeError:
                    missing_params.append(name)
                    if not param.is_meta:
                        new_state_dict[name] = param.data
            else:
                try:
                    hf_param = hf_model.get_parameter(name)
                    target_device = device if param.is_meta else param.device
                    new_state_dict[name] = hf_param.to(device=target_device, dtype=param.dtype)
                    loaded_params.add(name)
                except AttributeError:
                    if name.startswith('model.'):
                        try:
                            hf_param = hf_model.get_parameter(name[6:])
                            target_device = device if param.is_meta else param.device
                            new_state_dict[name] = hf_param.to(device=target_device, dtype=param.dtype)
                            loaded_params.add(name)
                        except AttributeError:
                            missing_params.append(name)
                            if not param.is_meta:
                                new_state_dict[name] = param.data
                    else:
                        missing_params.append(name)
                        if not param.is_meta:
                            new_state_dict[name] = param.data

    target_model.load_state_dict(new_state_dict, assign=True)

    for param in target_model.parameters():
        param.requires_grad_(False)

    if missing_params:
        print(f"Warning: Could not find {len(missing_params)} parameters in HF model:")
        for param in missing_params[:10]:
            print(f"  - {param}")
        if len(missing_params) > 10:
            print(f"  ... and {len(missing_params) - 10} more")

    meta_params = [name for name, param in target_model.named_parameters() if param.is_meta]
    if meta_params:
        raise RuntimeError(f"Failed to materialize {len(meta_params)} meta parameters")
