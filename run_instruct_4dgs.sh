#!/bin/bash

# ===================================================================
# ./run_instruct_4dgs.sh [dataset] [scene_name] [prompt] [guidance_scale] [image_guidance_scale]
# ===================================================================

if [ "$#" -lt 5 ] || [ "$#" -gt 6 ]; then
    echo "Usage: $0 <dataset> <scene_name> <prompt> <guidance_scale> <image_guidance_scale> [resize=512]"
    exit 1
fi

DATASET="$1"
SCENE_NAME="$2"
PROMPT="$3"
GUIDANCE_SCALE="$4"
IMAGE_GUIDANCE_SCALE="$5"
RESIZE="${6:-512}"

echo "------------------------------------------"
echo "  - dataset: ${DATASET}"
echo "  - scene: ${SCENE_NAME}"
echo "  - prompt: \"${PROMPT}\""
echo "  - resize: ${RESIZE}"
echo "------------------------------------------"
echo ""

echo "[1/4] Collect time0 images..."
python time0_collect.py --dataset ${DATASET} --scene_name ${SCENE_NAME}

echo ""

echo "[2/4] edit time0 images..."
python ./ip2p_models/multiview_edit.py \
    --dataset "${DATASET}" \
    --scene "${SCENE_NAME}" \
    --prompt "${PROMPT}" \
    --resize ${RESIZE} \
    --steps 20 \
    --guidance_scale ${GUIDANCE_SCALE} \
    --image_guidance_scale ${IMAGE_GUIDANCE_SCALE}

echo "✅ Completed time0 image editing."
echo ""

echo "[3/4] 3D editing"
python edit_3d.py \
    --configs "./arguments/${DATASET}/${SCENE_NAME}.py" \
    --ply_path "./output/${DATASET}/${SCENE_NAME}/point_cloud/iteration_14000/point_cloud.ply" \
    -s "./data/${DATASET}/${SCENE_NAME}" \
    --model_path "./output/${DATASET}/${SCENE_NAME}" \
    --dataset "${DATASET}" \
    --scene "${SCENE_NAME}" \
    --prompt "${PROMPT}" 

echo "✅ Completed 3d editing."
echo ""

echo "[4/4] Score refinement"
python refine_sds.py \
    --configs "./arguments/${DATASET}/${SCENE_NAME}.py" \
    --ply_path "./output/${DATASET}/${SCENE_NAME}/point_cloud_3dedit/${PROMPT}/iteration_1000/point_cloud.ply" \
    -s "./data/${DATASET}/${SCENE_NAME}" \
    --model_path "./output/${DATASET}/${SCENE_NAME}" \
    --prompt "${PROMPT}" \
    --guidance_scale ${GUIDANCE_SCALE} \
    --image_guidance_scale ${IMAGE_GUIDANCE_SCALE} \
    --resize ${RESIZE}

echo "✅ Completed score refinement."
echo ""

echo "🎉 All pipeline steps have been executed."