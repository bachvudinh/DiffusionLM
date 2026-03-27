"""
Model Weight Loading Utilities.

This module provides utilities for loading model weights from safetensors
files and HuggingFace models.

Based on JetEngine's loader.py:
https://github.com/Jet-Astra/SDAR/blob/main/jetengine/utils/loader.py

================================================================================
                              USAGE
================================================================================

    from vdllm.utils import load_model, load_from_hf_model

    # Load from local safetensors
    load_model(model, "/path/to/model/weights")

    # Load from HuggingFace model
    from transformers import AutoModel
    hf_model = AutoModel.from_pretrained("model_name")
    load_from_hf_model(target_model, hf_model)

"""

import os
from glob import glob
from typing import Optional

import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    """Default weight loader that directly copies data."""
    param.data.copy_(loaded_weight)


def _register_empty_parameter(module: nn.Module, name: str):
    """Register an empty parameter on meta device."""
    empty = nn.Parameter(torch.empty(0, device='meta'), requires_grad=False)
    module.register_parameter(name, empty)


def load_model(model: nn.Module, path: str) -> None:
    """
    Load model weights from safetensors files.

    Args:
        model: Target model to load weights into
        path: Directory containing safetensors files
    """
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})

    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, framework="pt", device="cpu") as f:
            names = [name for name in f.keys() if 'weight' in name or 'bias' in name]
            for weight_name in names:
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))


def load_from_hf_model(target_model: nn.Module, hf_model: nn.Module) -> None:
    """
    Load weights from a HuggingFace model to the target model.

    Handles fused QKV projections and gate_up_proj automatically.

    Args:
        target_model: Target model for inference engine
        hf_model: HuggingFace model to load from
    """
    device = next(hf_model.parameters()).device

    loaded_params = set()
    missing_params = []
    new_state_dict = {}

    for name, param in target_model.named_parameters():
        with torch.no_grad():
            # Handle fused qkv_proj
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

                    if param.is_meta:
                        new_state_dict[name] = qkv_weight.to(device=device, dtype=param.dtype)
                    else:
                        new_state_dict[name] = qkv_weight.to(device=param.device, dtype=param.dtype)
                    loaded_params.add(name)
                except AttributeError:
                    missing_params.append(name)
                    if not param.is_meta:
                        new_state_dict[name] = param.data

            # Handle fused gate_up_proj
            elif 'gate_up_proj.weight' in name and 'experts.' not in name:
                base_name = name.replace('gate_up_proj.weight', '')
                gate_name = base_name + 'gate_proj.weight'
                up_name = base_name + 'up_proj.weight'

                try:
                    gate_weight = hf_model.get_parameter(gate_name)
                    up_weight = hf_model.get_parameter(up_name)
                    gate_up_weight = torch.cat([gate_weight, up_weight], dim=0)

                    if param.is_meta:
                        new_state_dict[name] = gate_up_weight.to(device=device, dtype=param.dtype)
                    else:
                        new_state_dict[name] = gate_up_weight.to(device=param.device, dtype=param.dtype)
                    loaded_params.add(name)
                except AttributeError:
                    missing_params.append(name)
                    if not param.is_meta:
                        new_state_dict[name] = param.data

            # Handle regular parameters
            else:
                try:
                    hf_param = hf_model.get_parameter(name)
                    if param.is_meta:
                        new_state_dict[name] = hf_param.to(device=device, dtype=param.dtype)
                    else:
                        new_state_dict[name] = hf_param.to(device=param.device, dtype=param.dtype)
                    loaded_params.add(name)
                except AttributeError:
                    # Try without model prefix
                    if name.startswith('model.'):
                        try:
                            hf_param = hf_model.get_parameter(name[6:])
                            if param.is_meta:
                                new_state_dict[name] = hf_param.to(device=device, dtype=param.dtype)
                            else:
                                new_state_dict[name] = hf_param.to(device=param.device, dtype=param.dtype)
                            loaded_params.add(name)
                        except AttributeError:
                            missing_params.append(name)
                            if not param.is_meta:
                                new_state_dict[name] = param.data
                    else:
                        missing_params.append(name)
                        if not param.is_meta:
                            new_state_dict[name] = param.data

    # Load state dict
    target_model.load_state_dict(new_state_dict, strict=False)

    # Disable gradients
    for param in target_model.parameters():
        param.requires_grad_(False)

    # Report status
    if missing_params:
        print(f"Warning: Could not find {len(missing_params)} parameters in HF model:")
        for param in missing_params[:10]:
            print(f"  - {param}")
        if len(missing_params) > 10:
            print(f"  ... and {len(missing_params) - 10} more")

    # Check for meta parameters
    meta_params = [name for name, param in target_model.named_parameters() if param.is_meta]
    if meta_params:
        print(f"ERROR: {len(meta_params)} parameters still on meta device after loading!")
        raise RuntimeError(f"Failed to materialize {len(meta_params)} meta parameters")
