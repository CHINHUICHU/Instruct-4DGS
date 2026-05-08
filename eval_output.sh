#!/bin/bash

# ===================================================================
# ./eval_output.sh [dataset] [scene_name] [prompt]
# Evaluates the final refined output from run_instruct_4dgs.sh
# Computes PSNR, SSIM, LPIPS, and CLIP score via render_metric.py
# ===================================================================

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <dataset> <scene_name> <prompt>"
    echo "Example: $0 dynerf cook_spinach \"Make it look like a fauvism painting\""
    exit 1
fi

DATASET="$1"
SCENE_NAME="$2"
PROMPT="$3"

REFINE_DIR="./output/${DATASET}/${SCENE_NAME}/point_cloud_refine/${PROMPT}"

# Pick the highest iteration checkpoint
PLY_PATH=$(find "${REFINE_DIR}" -name "point_cloud.ply" | sort -t_ -k2 -n | tail -1)

if [ -z "${PLY_PATH}" ]; then
    echo "No refined point cloud found at: ${REFINE_DIR}"
    echo "Make sure run_instruct_4dgs.sh completed step 4."
    exit 1
fi

echo "------------------------------------------"
echo "  - dataset:  ${DATASET}"
echo "  - scene:    ${SCENE_NAME}"
echo "  - prompt:   \"${PROMPT}\""
echo "  - ply_path: ${PLY_PATH}"
echo "------------------------------------------"
echo ""

python render_metric.py \
    --configs "./arguments/${DATASET}/${SCENE_NAME}.py" \
    --ply_path "${PLY_PATH}" \
    --prompt "${PROMPT}" \
    -s "./data/${DATASET}/${SCENE_NAME}" \
    --model_path "./output/${DATASET}/${SCENE_NAME}"

echo ""
echo "Evaluation complete for: ./output/${DATASET}/${SCENE_NAME}/"
