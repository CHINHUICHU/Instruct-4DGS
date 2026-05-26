# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Always run Python commands inside the `gaussian_splatting_cu12` conda environment:
```bash
source /home/chinhui/miniforge3/bin/activate gaussian_splatting_cu12 && python ...
```

Set this env var before running training or editing to avoid CUDA OOM fragmentation:
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## Key Commands

**Full editing pipeline** (4 steps):
```bash
# run_instruct_4dgs.sh [dataset] [scene_name] [prompt] [guidance_scale] [image_guidance_scale] [resize]
bash run_instruct_4dgs.sh dynerf cook_spinach "Make it look like a fauvism painting" 10.5 1.2
```

**Train initial 4DGS** (prerequisite; follow upstream 4DGS repo):
```bash
python train.py --configs ./arguments/{dataset}/{scene}.py -s ./data/{dataset}/{scene} --model_path ./output/{dataset}/{scene} --expname {expname}
```

**Render edited scene**:
```bash
bash render_output.sh dynerf cook_spinach "Make it look like a fauvism painting"
# or directly:
python render_edited4d.py --configs ./arguments/dynerf/cook_spinach.py \
    --ply_path "./output/dynerf/cook_spinach/point_cloud_refine/Make it look like a fauvism painting/iteration_800/point_cloud.ply" \
    -s ./data/dynerf/cook_spinach --model_path ./output/dynerf/cook_spinach
```

**Evaluate (PSNR, SSIM, LPIPS, CLIP)**:
```bash
bash eval_output.sh dynerf cook_spinach "Make it look like a fauvism painting"
```

## Architecture

Instruct-4DGS (CVPR 2025) edits dynamic 3D scenes by separating static and dynamic components:

**4D Gaussian representation** (`scene/gaussian_model.py`):
- Static canonical 3D Gaussians (`_xyz`, `_features_dc`, `_features_rest`, `_scaling`, `_rotation`, `_opacity`)
- A HexPlane-based deformation field (`_deformation`) that warps the static Gaussians over time
- `scene/deformation.py` implements the `Deformation` MLP on top of `scene/hexplane.py`'s `HexPlaneField`

**4-step editing pipeline** (orchestrated by `run_instruct_4dgs.sh`):
1. **`time0_collect.py`** — copies the first frame (`0000.png`) of each camera view from `./data/{dataset}/{scene}/cam{N}/images/` into `./data/{dataset}/time0_{scene}/original_time0_{N}.png`
2. **`ip2p_models/multiview_edit.py`** — applies InstructPix2Pix (using `timbrooks/instruct-pix2pix` via a custom `UNet3DConditionModel`) to the t=0 multi-view images; saves edited images as `edited_{last_word_of_prompt}_original_time0_{N}.png` in the same folder
3. **`edit_3d.py`** — fits only the static Gaussians to the edited t=0 images (1000 iterations); loads a pre-trained 4DGS `.ply` and its deformation model, then optimizes with `training_only3dgs_setup`; saves to `output/{dataset}/{scene}/point_cloud_3dedit/{prompt}/`
4. **`refine_sds.py`** — runs score distillation sampling (SDS) using IP2P to fix misalignment between the edited static Gaussians and the deformation field; saves to `output/{dataset}/{scene}/point_cloud_refine/{prompt}/`

**Scene configs**: `arguments/{dataset}/{scene}.py` — per-scene overrides for `ModelParams`, `OptimizationParams`, `PipelineParams`, `ModelHiddenParams`. The `ModelHiddenParams` controls the deformation network (HexPlane resolution, MLP width/depth, TV losses).

**Dataset support**: `scene/__init__.py` auto-detects the dataset type from directory structure:
- `sparse/` → COLMAP
- `transforms_train.json` → Blender/D-NeRF
- `poses_bounds.npy` → Dynerf
- `dataset.json` → Nerfies/HyperNeRF
- `train_meta.json` → PanopticSports

## Important Caveats

**Missing cameras in Dynerf**: Some Dynerf scenes have non-contiguous camera indices. `edit_3d.py` contains hardcoded remapping dicts (`dict_coffee_martini`, `dict_sear_steak`) at lines 46–88 to map training iteration numbers to camera indices. For scenes other than `sear_steak`, `coffee_martini`, and `cook_spinach` with `maxtime < 6000`, the code raises `NotImplementedError` — you must add a new dict.

**Edited image path convention**: `edit_3d.py` looks for edited images at `./data/{dataset}/{scene}/{last_word_of_prompt}/edited_{last_word_of_prompt}_original_time0_{N}.png`. The last word of the prompt (punctuation stripped) determines the subfolder name.

**Submodules**: `submodules/diff-gaussian-rasterization` and `submodules/simple-knn` are C++ CUDA extensions that must be compiled (`pip install -e ./submodules/...`) when setting up the environment.

**GaussCtrl integration** (`gaussctrl_models/`, `gaussctrl/`): An alternative multi-view consistent editing backend (ECCV 2024 GaussCtrl). This is a git submodule and uses a separate nerfstudio-based environment.
