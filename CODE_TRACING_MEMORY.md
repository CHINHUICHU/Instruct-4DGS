# Code Tracing Guide for Memory Optimization in Instruct-4DGS

## Pipeline Overview

The full pipeline runs in 4 sequential stages via `run_instruct_4dgs.sh`:

```
[1] time0_collect.py          → collect reference frames at t=0
[2] ip2p_models/multiview_edit.py  → edit images with InstructPix2Pix
[3] edit_3d.py                → 3D lifting: fit Gaussians to edited images
[4] refine_sds.py             → score distillation refinement using IP2P
```

The two memory-intensive scripts are **`edit_3d.py`** (step 3) and **`refine_sds.py`** (step 4). Both share the same training loop structure.

---

## Entry Points

### `edit_3d.py` — 3D Edit Loop
```
main()
└── training()                          edit_3d.py:~270
    ├── GaussianModel(sh_degree, hyper) scene/gaussian_model.py:47
    ├── Scene(dataset, gaussians)        scene/__init__.py
    ├── gaussians.load_ply(ply_path)     scene/gaussian_model.py:294
    ├── gaussians.load_model(...)        scene/gaussian_model.py:253
    └── scene_reconstruction(...)        edit_3d.py:~90
```

### `refine_sds.py` — SDS Refinement Loop
```
main()
└── training()                          refine_sds.py:406
    ├── [same gaussian setup as above]
    ├── AutoencoderKL.from_pretrained()  (VAE on GPU, fp16)
    ├── CLIPTextModel.from_pretrained()  (text encoder on GPU, fp16)
    ├── UNet3DConditionModel             (main diffusion model on GPU, fp16)
    └── scene_reconstruction(...)        refine_sds.py:~77
```

---

## Memory Allocation Map

### Gaussian Model (`scene/gaussian_model.py`)

| Tensor | Description | Allocated at |
|--------|-------------|--------------|
| `_xyz` | Gaussian centers `[N, 3]` | `load_ply()` line 294 |
| `_features_dc` | DC color SH `[N, 1, 3]` | `load_ply()` |
| `_features_rest` | Higher-order SH `[N, 15, 3]` | `load_ply()` |
| `_scaling` | Scale `[N, 3]` | `load_ply()` |
| `_rotation` | Quaternion `[N, 4]` | `load_ply()` |
| `_opacity` | Opacity `[N, 1]` | `load_ply()` |
| `_deformation` | HexPlane + MLP network | `GaussianModel.__init__()` line 57 |
| `_deformation_table` | Bool mask `[N]` | set in `training()` |
| `max_radii2D` | For densification `[N]` | during training |
| optimizer states | Adam first/second moments (×2 per param) | `training_only3dgs_setup()` line 166 |

**Hotspot:** With N=360,000 Gaussians, all per-point tensors plus their Adam optimizer states sit on GPU simultaneously.

### Deformation Network (`scene/deformation.py` + `scene/hexplane.py`)

```
Deformation.__init__()
├── HexPlaneField (6 feature planes over (x,y,z,t) combinations)
│   └── Each plane: nn.Parameter of shape [1, feat_dim, reso_i, reso_j]
│       Stored as fp32 on GPU                   hexplane.py:~60
├── feature_out: MLP [grid_out_dim → W] × D layers  deformation.py:52
├── pos_deform, scales_deform, rotations_deform,
│   opacity_deform, shs_deform: small MLPs          deformation.py:57-62
└── (optional) empty_voxel: DenseGrid [64,64,64]    deformation.py:26
```

### Diffusion Models — `refine_sds.py` only

```
training()
├── vae        → fp16, GPU    refine_sds.py:437
├── text_encoder → fp16, GPU  refine_sds.py:438
└── unet       → fp16, GPU    refine_sds.py:439
```
These three models stay resident on GPU for the entire training run.

---

## Training Loop Trace — Per Iteration

Both scripts share this pattern. Trace the call stack from `scene_reconstruction()`:

### Step 1: Camera Batch
```python
# edit_3d.py:~140 / refine_sds.py:~140
viewpoint_cams = [...]   # batch_size cameras from DataLoader or stack
```

### Step 2: Render
```python
# edit_3d.py:~165 / refine_sds.py:~165
render_pkg = render(viewpoint_cam, gaussians, pipe, background, stage="fine")
#   └── gaussian_renderer/__init__.py: render()
#         ├── screenspace_points = torch.zeros_like(pc.get_xyz, requires_grad=True)
#         │     ← [N, 3] gradient-tracked tensor created EVERY iteration
#         ├── pc._deformation(means3D, scales, rotations, opacity, shs, time)
#         │     └── deformation.py: forward_dynamic()
#         │           └── query_time() → HexPlaneField.forward() [grid_sample calls]
#         │                 → feature_out MLP → pos_deform/scales_deform/...
#         └── GaussianRasterizer(...)  ← CUDA kernel, outputs rendered image
```

**Key allocation:** `screenspace_points` is a new `[N, 3]` CUDA tensor with `requires_grad=True` every single iteration — this is kept alive for backward pass gradient accumulation.

### Step 3: Loss + Backward

**`edit_3d.py` path** (L1 against pre-edited images):
```python
# edit_3d.py:~200
gt_image = Image.open(edited_images_path/...)  # loaded from disk each iter
Ll1 = l1_loss(image_tensor, gt_image_tensor)
loss.backward()
```

**`refine_sds.py` path** (SDS loss through VAE + UNet):
```python
# refine_sds.py:~248-315
vae_input_images = F.interpolate(image_tensor, ...)  # resized rendered images
latents = encode_1(ip2p, ...)                         # VAE encode, fp16
image_latents = encode_2(ip2p, ...)                   # VAE encode (mode), fp16
# ... noise prediction through UNet3DConditionModel
# sequence_length=4 frames stacked → b×4 latent volume through unet
loss_sds = 0.5 * F.mse_loss(noise_pred, target, ...)
loss_sds.backward()
```
This is the primary VRAM spike point: VAE + UNet forward pass runs **inside** the training loop with Gaussian activations still live in memory.

### Step 4: Gradient Accumulation + Optimizer
```python
# both scripts
viewspace_point_tensor_grad = sum(viewspace_point_tensor_list[i].grad)
gaussians.optimizer.step()         # updates all Gaussian parameters
gaussians.optimizer.zero_grad()
```

### Step 5: Densification / Pruning (during first ~15k iters)
```python
# edit_3d.py:~300
gaussians.densify(...)   # gaussian_model.py:521 → densify_and_split() + densify_and_clone()
gaussians.prune(...)     # gaussian_model.py:509
gaussians.grow(...)      # gaussian_model.py:494 → add_point_by_mask()
```
**Peak memory moment:** densification clones tensors before pruning — temporarily doubles per-point tensor sizes.

---

## Key Files for Memory Investigation

```
scene/gaussian_model.py      All Gaussian parameters and optimizer setup
scene/deformation.py         HexPlane + MLP deformation network
scene/hexplane.py            Feature grid definition (grid_sample calls)
gaussian_renderer/__init__.py render() — screenspace_points allocation
edit_3d.py                   Step 3 training loop
refine_sds.py                Step 4 training loop (also loads VAE/CLIP/UNet)
arguments/                   Per-scene config files (batch_size, iterations, etc.)
```

---

## How to Instrument for Memory Profiling

### Quick GPU snapshot (insert anywhere in training loop)
```python
import torch
def mem_snapshot(tag=""):
    alloc = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"[MEM {tag}] alloc={alloc:.2f}GB reserved={reserved:.2f}GB")
```
Insert before/after `render()`, before/after `loss.backward()`, and before/after `optimizer.step()`.

### PyTorch memory snapshot (full allocation trace)
```python
# Add before training loop starts
torch.cuda.memory._record_memory_history(max_entries=100000)

# ... run a few iterations ...

torch.cuda.memory._dump_snapshot("mem_snapshot.pkl")
torch.cuda.memory._record_memory_history(enabled=None)
# Visualize: python -m torch.cuda.memory_viz mem_snapshot.pkl
```

### Run with memory profiler
```bash
source /home/chinhui/miniforge3/bin/activate gaussian_splatting_cu12

# Profile step 3 (edit_3d):
python -m torch.utils.bottleneck edit_3d.py \
    --configs "./arguments/dynerf/sear_steak.py" \
    --ply_path "./output/dynerf/sear_steak/point_cloud/iteration_14000/point_cloud.ply" \
    -s "./data/dynerf/sear_steak" \
    --model_path "./output/dynerf/sear_steak" \
    --dataset dynerf --scene sear_steak --prompt "your prompt"

# Watch live GPU usage
watch -n 0.5 nvidia-smi
```

---

## Memory Optimization Hooks to Investigate

### 1. `screenspace_points` — created every iteration
- **Location:** `gaussian_renderer/__init__.py:26`
- **Issue:** `torch.zeros_like(..., requires_grad=True)` allocates `[N, 3]` fp32 = ~4MB at N=360k, every iteration
- **Idea:** reuse a pre-allocated buffer

### 2. `encode_2` (image conditioning) runs with gradient tracking — `refine_sds.py:256`
- **Fix:** wrap only `encode_2` in `torch.no_grad()`. `encode_1` must stay differentiable because `latents → loss_sds.backward()` flows gradients back through the VAE encoder to the rendered images and then to the Gaussians. `encode_2` produces `image_latents` used only as UNet conditioning inside the existing `torch.no_grad()` block, so it never needs a grad graph.

### 3. GT image loading in `edit_3d.py`
- **Location:** `edit_3d.py:~195`
- **Issue:** `Image.open()` → `transform()` → `.cuda()` every iteration
- **Idea:** pre-cache all edited images in a dict or RAM at the start of training

### 3. Diffusion models live alongside Gaussians in `refine_sds.py`
- **Location:** `refine_sds.py:437-439`
- **Issue:** VAE + text encoder + UNet all resident in GPU fp16 throughout training
- **Ideas:** CPU offload text encoder (used only once for prompt embed), use `torch.autocast` more aggressively, offload VAE between iterations

### 4. Densification temporarily doubles per-point tensors
- **Location:** `gaussian_model.py:435 densify_and_split()`, `gaussian_model.py:387 cat_tensors_to_optimizer()`
- **Issue:** concatenates new tensors to optimizer state before pruning old ones
- **Idea:** prune first, then clone, or run `torch.cuda.empty_cache()` after prune step

### 5. HexPlane grid resolution
- **Location:** `scene/hexplane.py:60`, configured via `kplanes_config` in `arguments/*.py`
- **Issue:** 6 feature planes, each stored as fp32 parameter
- **Ideas:** quantize planes to fp16, use gradient checkpointing on `feature_out` MLP

### 6. SDS latent batch size
- **Location:** `refine_sds.py:82` — `sequence_length = 4`
- **Issue:** 4 frames processed as a batch through VAE and UNet each iteration
- **Idea:** reduce `sequence_length`, accumulate gradients across sub-batches

### 7. Optimizer state memory
- **Location:** `gaussian_model.py:166 training_only3dgs_setup()`
- **Issue:** Adam stores 2× the parameter memory (first + second moment)
- **Ideas:** use 8-bit Adam (`bitsandbytes`), paged Adam, or Adan

---

## Activation Checkpoint Candidate Calls

For `refine_sds.py`, the UNet forward pass is the largest single activation memory block. Apply `torch.utils.checkpoint.checkpoint` to transformer blocks inside `ip2p_models/models/ip2p_unet.py` (each `UNet3DConditionOutput` block is a natural boundary).

For the deformation network, apply checkpoint to `Deformation.forward_dynamic()` in `scene/deformation.py:~100` — the hidden state is `[N, W]` which at N=360k, W=256 is ~375MB fp32.

---

## Quick Sanity Numbers

| Component | Approx VRAM at N=360k |
|-----------|----------------------|
| `_xyz` + grad | 8 MB |
| `_features_dc/rest` | ~50 MB |
| `_scaling/rotation/opacity` | ~20 MB |
| Adam states (all Gaussians) | ×2 of above |
| Deformation network (HexPlane+MLP) | 200–500 MB (config-dependent) |
| VAE fp16 (refine_sds only) | ~350 MB |
| CLIP text encoder fp16 (refine_sds only) | ~300 MB |
| UNet3D fp16 (refine_sds only) | ~3–6 GB |
| Per-iteration activations (SDS latents) | ~500 MB spike |
