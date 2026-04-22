#!/bin/bash

# ===================================================================
# ./render_output.sh [dataset] [scene_name] [prompt]
# Renders the final refined output from run_instruct_4dgs.sh
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

python render_edited4d.py \
    --configs "./arguments/${DATASET}/${SCENE_NAME}.py" \
    --ply_path "${PLY_PATH}" \
    -s "./data/${DATASET}/${SCENE_NAME}" \
    --model_path "./output/${DATASET}/${SCENE_NAME}"

echo ""
echo "Renders saved to: ./output/${DATASET}/${SCENE_NAME}/"
