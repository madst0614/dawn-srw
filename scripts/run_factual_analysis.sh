#!/bin/bash
# Factual Knowledge Neuron Analysis

CHECKPOINT=${1:-"checkpoint.pt"}
OUTPUT_DIR=${2:-"routing_analysis/factual"}
ITERATIONS=${3:-100}
LAYER=${4:-11}
POOL=${5:-"fv"}

echo "========================================"
echo "Factual Knowledge Neuron Analysis"
echo "========================================"
echo "Checkpoint: $CHECKPOINT"
echo "Output: $OUTPUT_DIR"
echo "Iterations: $ITERATIONS"
echo "Layer: $LAYER"
echo "Pool: $POOL"
echo "========================================"

mkdir -p "$OUTPUT_DIR"

declare -a PROMPTS=(
    "The capital of France is"
    "The capital of Germany is"
    "The capital of Japan is"
    "The color of the sky is"
)

declare -a TARGETS=(
    "paris"
    "berlin"
    "tokyo"
    "blue"
)

declare -a NAMES=(
    "france_capital"
    "germany_capital"
    "japan_capital"
    "sky_color"
)

for i in "${!PROMPTS[@]}"; do
    PROMPT="${PROMPTS[$i]}"
    TARGET="${TARGETS[$i]}"
    NAME="${NAMES[$i]}"

    echo ""
    echo "[$((i+1))/${#PROMPTS[@]}] $NAME"
    echo "Prompt: '$PROMPT'"
    echo "Target: '$TARGET'"

    python scripts/analysis/standalone/routing_analysis.py \
        --checkpoint "$CHECKPOINT" \
        --prompt "$PROMPT" \
        --target_token "$TARGET" \
        --iterations "$ITERATIONS" \
        --layer "$LAYER" \
        --pool "$POOL" \
        --bf16 \
        --output "$OUTPUT_DIR"
done

echo ""
echo "Done! Results in: $OUTPUT_DIR"
