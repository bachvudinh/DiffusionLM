#!/bin/bash
# Metal GPU profiler for MLX inference
# Usage: ./metal_profiler.sh [prompt] [model_path]
#
# This will:
# 1. Enable Metal GPU capture (MTL_CAPTURE_ENABLED=1)
# 2. Run the vdllm profiler with real model
# 3. Save trace to a .gputrace file
# 4. Auto-open the trace in Xcode

set -e

MODEL_PATH="${2:-/tmp/sdar-1.7b-chat}"
PROMPT="${1:-Hello world}"
TRACE_FILE="$(pwd)/vdllm_trace_$(date +%Y%m%d_%H%M%S).gputrace"

echo "=============================================="
echo "MLX Metal Profiler"
echo "=============================================="
echo "Model: $MODEL_PATH"
echo "Prompt: $PROMPT"
echo "Trace file: $TRACE_FILE"
echo ""

# Enable Metal capture and run profiler
MTL_CAPTURE_ENABLED=1 \
    python -m vdllm.profiling.mlx_profiler \
        --model "$MODEL_PATH" \
        --prompt "$PROMPT" \
        --gen-length 256 \
        --block-length 4 \
        --denoising-steps 4 \
        --runs 3 \
        --chat \
        --capture "$TRACE_FILE"

echo ""
echo "=============================================="
echo "Opening trace in Xcode..."
open -a Xcode "$TRACE_FILE"
echo ""
echo "In Xcode GPU debugger you can:"
echo "  - See timeline of all GPU operations"
echo "  - Identify kernel bottlenecks"
echo "  - Analyze memory transfers"
echo "  - See GPU utilization"
echo "=============================================="
