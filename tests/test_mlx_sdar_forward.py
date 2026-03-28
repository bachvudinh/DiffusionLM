"""
Validates MLX SDAR forward pass matches PyTorch HuggingFace reference.
Compares logits on a test prompt: top-1 match, top-5 overlap, cosine similarity.
"""

import sys
import os
import time
import tempfile
import shutil
from types import ModuleType
from unittest.mock import MagicMock

import torch
import mlx.core as mx
import numpy as np

# Disable torch.compile
import torch._dynamo
torch._dynamo.config.disable = True

WORK_DIR = "/Users/bachvu/Workplace/vibe-space/DLM-playground/DiffusionLM"
MODEL_PATH = "/tmp/sdar-1.7b-chat"

sys.path.insert(0, WORK_DIR)

# ============================================================================
# PATCH: Mock flash_attn for CPU-only PyTorch
# ============================================================================

flash_attn_module = ModuleType('flash_attn')
flash_attn_module.__spec__ = MagicMock()
flash_attn_module.__file__ = "/dev/null"


def _rms_norm_fn(hidden_states, weight=None, bias=None, eps=1e-6):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdims=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return (weight * hidden_states).to(input_dtype)


ops_module = ModuleType('flash_attn.ops')
ops_module.__spec__ = MagicMock()
ops_module.__file__ = "/dev/null"
ops_module.layer_norm = ModuleType('flash_attn.ops.triton.layer_norm')
ops_module.layer_norm.__spec__ = MagicMock()
ops_module.layer_norm.rms_norm_fn = _rms_norm_fn

bert_padding_module = ModuleType('flash_attn.bert_padding')
bert_padding_module.__spec__ = MagicMock()
bert_padding_module.index_first_axis = lambda x, indices: torch.index_select(x.view(-1, x.shape[-1]), 0, indices)
bert_padding_module.pad_input = lambda x, indices, batch_size, seq_len: x
bert_padding_module.unpad_input = lambda x, attention_mask: (x, None, None)

flash_attn_module.ops = ops_module
flash_attn_module.bert_padding = bert_padding_module
flash_attn_module.flash_attn_func = None
flash_attn_module.flash_attn_varlen_func = None

sys.modules['flash_attn'] = flash_attn_module
sys.modules['flash_attn.ops'] = ops_module
sys.modules['flash_attn.ops.triton'] = ModuleType('flash_attn.ops.triton')
sys.modules['flash_attn.ops.triton'].__spec__ = MagicMock()
sys.modules['flash_attn.ops.triton.layer_norm'] = ops_module.layer_norm
sys.modules['flash_attn.bert_padding'] = bert_padding_module

# ============================================================================
# PATCH: Fix transformers 5.x compatibility
# ============================================================================
import transformers.cache_utils as cache_utils
from transformers.cache_utils import Cache, DynamicCache, StaticCache


class SlidingWindowCache(DynamicCache):
    pass


cache_utils.SlidingWindowCache = SlidingWindowCache

# Create temp package for HF model
TEMP_PKG = tempfile.mkdtemp(prefix="sdar_pkg_")
MODEL_PKG = os.path.join(TEMP_PKG, "sdar_model")
os.makedirs(MODEL_PKG)

for f in ["configuration_sdar.py", "modeling_sdar.py", "tokenization_qwen2.py", "tokenization_qwen2_fast.py"]:
    src = os.path.join(MODEL_PATH, f)
    if os.path.exists(src):
        os.symlink(src, os.path.join(MODEL_PKG, f))

with open(os.path.join(MODEL_PKG, "__init__.py"), "w") as f:
    f.write("# SDAR model package\n")

with open(os.path.join(MODEL_PKG, "fused_linear_diffusion_cross_entropy.py"), "w") as f:
    f.write("class FusedLinearDiffusionCrossEntropyLoss: pass")

sys.path.insert(0, TEMP_PKG)

print("=" * 70)
print("SDAR MLX Forward Pass Test - All 28 Layers")
print("=" * 70)

# ============================================================================
# Load tokenizer
# ============================================================================
print("\n[1] Loading tokenizer...")
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
print(f"  Vocab size: {tokenizer.vocab_size}")

# ============================================================================
# Load PyTorch reference model
# ============================================================================
print("\n[2] Loading PyTorch reference model...")

from sdar_model.configuration_sdar import SDARConfig
from sdar_model.modeling_sdar import SDARForCausalLM as PTSDARForCausalLM

model_config = SDARConfig.from_pretrained(MODEL_PATH)
model_config.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id else 0
model_config.bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id else 1
model_config.eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id else 2
model_config._attn_implementation = 'eager'

if model_config.rope_scaling is not None and model_config.rope_scaling.get('rope_type') == 'default':
    model_config.rope_scaling['rope_type'] = 'linear'
    if 'factor' not in model_config.rope_scaling:
        model_config.rope_scaling['factor'] = 1.0

pt_model = PTSDARForCausalLM(model_config)

from safetensors.torch import load_file
state_dict = load_file(f"{MODEL_PATH}/model.safetensors")
pt_model.load_state_dict(state_dict, strict=False)
pt_model.eval()
print("  PyTorch model loaded successfully")

# ============================================================================
# Load MLX model
# ============================================================================
print("\n[3] Loading MLX SDAR model...")

from vdllm.models.mlx_sdar import SDARForCausalLM, SDARModelArgs

import json
from pathlib import Path

with open(f"{MODEL_PATH}/config.json") as f:
    config = json.load(f)

args = SDARModelArgs.from_dict(config)
print(f"  Config: {args.num_hidden_layers} layers, {args.num_attention_heads} heads, "
      f"{args.num_key_value_heads} KV heads, hidden={args.hidden_size}")

mlx_model = SDARForCausalLM(args)

weight_files = sorted(Path(MODEL_PATH).glob("*.safetensors"))
weights = {}
for wf in weight_files:
    weights.update(mx.load(str(wf)))
print(f"  Loaded {len(weights)} weight tensors")

weights = mlx_model.sanitize(weights)
mlx_model.load_weights(list(weights.items()), strict=False)
mx.eval(mlx_model.parameters())
print("  MLX model loaded successfully")

# ============================================================================
# Test prompt
# ============================================================================
prompt = "The capital of France is"
input_ids = tokenizer.encode(prompt, return_tensors='pt')
print(f"\n[4] Test prompt: '{prompt}'")
print(f"  Input tokens: {input_ids.tolist()}")
print(f"  Token count: {input_ids.shape[1]}")

# ============================================================================
# PyTorch reference (CPU)
# ============================================================================
print("\n[5] Running PyTorch reference forward...")
with torch.no_grad():
    t0 = time.time()
    pt_outputs = pt_model(input_ids)
    pt_logits = pt_outputs.logits
    t1 = time.time()
    print(f"  PyTorch time: {t1-t0:.3f}s")

pt_last = pt_logits[0, -1].numpy()
pt_top5 = np.argsort(pt_last)[-5:][::-1]

print(f"  PyTorch logits range: [{pt_last.min():.2f}, {pt_last.max():.2f}]")
print(f"  PyTorch top 5:")
for i, tok_id in enumerate(pt_top5):
    tok_str = tokenizer.decode([tok_id])
    print(f"    {i}: {tok_id:6d} ({pt_last[tok_id]:.4f}) -> '{tok_str}'")

# ============================================================================
# MLX forward pass
# ============================================================================
print("\n[6] Running MLX forward pass...")

mlx_input = mx.array(input_ids.numpy())

t0 = time.time()
mlx_logits = mlx_model(mlx_input)
mx.eval(mlx_logits)
t1 = time.time()
print(f"  MLX time: {t1-t0:.3f}s")

mlx_logits_np = mlx_logits.astype(mx.float32)
mlx_last = np.array(mlx_logits_np[0, -1])
mlx_top5 = np.argsort(mlx_last)[-5:][::-1]

print(f"  MLX logits range: [{mlx_last.min():.2f}, {mlx_last.max():.2f}]")
print(f"  MLX top 5:")
for i, tok_id in enumerate(mlx_top5):
    tok_str = tokenizer.decode([tok_id])
    print(f"    {i}: {tok_id:6d} ({mlx_last[tok_id]:.4f}) -> '{tok_str}'")

# ============================================================================
# Comparison
# ============================================================================
print("\n[7] Comparison:")
pt_top = int(pt_top5[0])
mlx_top = int(mlx_top5[0])
top_match = pt_top == mlx_top
print(f"  Top token match: {top_match} (PT={pt_top}, MLX={mlx_top})")

cos_sim = np.dot(pt_last, mlx_last) / (np.linalg.norm(pt_last) * np.linalg.norm(mlx_last))
print(f"  Cosine similarity: {cos_sim:.6f}")

pt_top5_set = set(pt_top5)
mlx_top5_set = set(mlx_top5)
top5_overlap = len(pt_top5_set & mlx_top5_set)
print(f"  Top-5 overlap: {top5_overlap}/5")

# ============================================================================
# Cleanup
# ============================================================================
print(f"\nCleaning up temporary package: {TEMP_PKG}")
shutil.rmtree(TEMP_PKG)

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
success = cos_sim > 0.90 and top_match and top5_overlap >= 4
if success:
    print("SUCCESS! MLX matches PyTorch reference")
    print(f"  - Top token: {'MATCH' if top_match else 'MISMATCH'}")
    print(f"  - Cosine similarity: {cos_sim:.6f} (> 0.90)")
    print(f"  - Top-5 overlap: {top5_overlap}/5")
else:
    print("FAILURE! MLX does NOT match PyTorch reference")
    print(f"  - Top token: {'MATCH' if top_match else 'MISMATCH'}")
    print(f"  - Cosine similarity: {cos_sim:.6f} (should be > 0.90)")
    print(f"  - Top-5 overlap: {top5_overlap}/5")
print("=" * 70)
