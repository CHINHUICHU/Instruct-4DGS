# Running with the Ex4DGS backbone

```bash
git submodule update --init --recursive Ex4DGS
```

## Editing pipeline

Steps 1–2 (t=0 collection + IP2P multi-view edit) are shared with the native pipeline and produce the
edited t=0 images under `data/dynerf/cook_spinach/{last_word_of_prompt}/`. Then run the Ex4DGS stages
(pretrained model at `Ex4DGS/output/point_cloud/iteration_40000`):

```bash
PROMPT="Make it look like a fauvism painting"
SRC=/Instruct-4DGS/data/dynerf/cook_spinach

# Step 3 — fit static Gaussians to edited t=0 images
python edit_ex4dgs.py --model_path output/ --source_path "$SRC" --loader dynerf \
    --prompt "$PROMPT" --edit_iters 1000

# Step 4 — SDS refinement
python refine_sds_ex4dgs.py --model_path output/ --source_path "$SRC" --loader dynerf \
    --ply_path "output/point_cloud_edit/$PROMPT/point_cloud.ply" --prompt "$PROMPT" \
    --guidance_scale 10.5 --image_guidance_scale 1.2 --sds_iters 800 --resize 512

# Step 5 — render + metrics (PSNR/SSIM/LPIPS/CLIP)
python render_metric_ex4dgs.py --model_path output/ --source_path "$SRC" --loader dynerf \
    --ply_path "output/point_cloud_refine/$PROMPT/point_cloud.ply" --prompt "$PROMPT"
```

## Video-quality evaluation

Run from the repo root in the `gaussian_splatting_cu12` env:

```bash
# Edited vs original: PSNR/SSIM/LPIPS + sharpness/flow/CLIP metrics
python evaluation.py --original_video <orig.mp4|frames_dir> --edited_video <edited.mp4|frames_dir> \
    --prompt "a video of a man cooking spinach in fauvism style" --output out.json

# Reconstruction quality of a render vs GT
python evaluate_video_quality.py --render_video <render.mp4> --gt_video <gt.mp4> --output out.json
```
