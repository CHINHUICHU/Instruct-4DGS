# Instruct-4DGS Baseline Results — cook_spinach

Baseline measurement of the **original Instruct-4DGS** (4DGS + HexPlane deformation) pipeline, to be compared later against the 4D-Scaffold-GS variant. Date: 2026-05-24.

## Setup

| Item | Value |
|---|---|
| Dataset / scene | dynerf (N3DV) / `cook_spinach` |
| Pre-trained 4DGS checkpoint | `output/dynerf/cook_spinach/point_cloud/iteration_14000` |
| Pipeline | `run_instruct_4dgs.sh` (time0_collect → multiview_edit → edit_3d → refine_sds) |
| IP2P steps | 20 |
| Guidance scale (text / image) | **10.5 / 1.2** (repo default) |
| edit_3d iterations | 1000 |
| refine_sds iterations | 800 |
| Resolution | original (no `--resize`) |
| GPU | 24 GB (RTX 4090-class) |
| Env | conda `gaussian_splatting_cu12`, torch 2.4.1+cu121, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |

Prompts evaluated:
- **Fauvism** — `Make it look like a fauvism painting` (global style; reused existing outputs)
- **Sculpture** — `Turn it into a marble sculpture` (global style)
- **Man→woman** — `Turn the man into a woman` (local / semantic)

## Memory usage (peak VRAM)

Measured via `torch.cuda.max_memory_allocated()` / `max_memory_reserved()` over each training stage. **Peak VRAM is prompt-independent** (same checkpoint, scene, and ~100k Gaussians) — confirmed: sculpture and man→woman differ by <0.01 GB.

| Stage | Peak allocated | Peak reserved | Gaussians |
|---|---|---|---|
| `edit_3d` (fine) | **2.40 GB** | 2.49 GB | ~100k–104k |
| `refine_sds` (fine) | **6.92 GB** | 6.97 GB | ~100k–102k |

- The full pipeline peaks at **~7 GB** — well within the 24 GB card at default resolution.
- `refine_sds` dominates because the IP2P UNet + VAE + text encoder are GPU-resident alongside the Gaussian optimization; `edit_3d` (plain L1 fitting) is ~3× lighter.
- Earlier OOM incidents were not reproduced here — likely tied to higher resolution or a heavier scene, not this config.

### Track B attempt: UNet CPU-offload (negative result)

Tried keeping the IP2P UNet on CPU and moving it to GPU only for the `no_grad` noise-prediction step (`refine_sds.py --offload_unet`, sculpture prompt):

| refine_sds | Peak allocated | Peak reserved | Runtime |
|---|---|---|---|
| Baseline | 6.92 GB | 6.97 GB | ~4 min |
| + UNet offload | 6.72 GB | 6.78 GB | ~7 min |

**Only 0.20 GB saved at ~75% longer runtime — not worth it.** Peak VRAM occurs *during the UNet forward pass*, the one window where the UNet must be GPU-resident; at that instant the GPU also holds the Gaussian params+optimizer, the retained differentiable VAE-encode graph (`encode_1`), VAE weights, and latents. Offloading the UNet during the idle/backward window doesn't lower the high-water mark. To actually cut the refine_sds peak, reduce **render resolution** or **`sequence_length`**, not weight residency. The `--offload_unet` flag remains (default off); baseline behavior unchanged without it.

## Quality metrics

Computed by `render_metric.py` over one frame per camera view.

> **Interpretation (important):** PSNR / SSIM / LPIPS are measured against the **original (unedited) training images**, so they quantify *preservation* — higher PSNR/SSIM = closer to the original = milder edit. **CLIP** is the rendered frame vs the **edit prompt** — higher = stronger prompt alignment. The two axes trade off.

| Prompt | Type | SSIM ↑* | PSNR ↑* | LPIPS-VGG ↓* | LPIPS-Alex ↓* | CLIP ↑ |
|---|---|---|---|---|---|---|
| Fauvism painting | global style | 0.5636 | 12.41 | 0.5696 | 0.4071 | **0.2861** |
| Marble sculpture | global style | 0.6892 | 13.21 | 0.4208 | 0.3272 | 0.2364 |
| Man → woman | local/semantic | **0.8401** | **23.93** | **0.2523** | **0.1157** | 0.2334 |

\* vs. original images — these measure *deviation from the original scene*, not "edit quality." For an edit, lower PSNR/SSIM means a stronger visual change.

### Reading the results

- **Fauvism** = strongest edit: most deviation from the original (lowest SSIM/PSNR) and best prompt alignment (highest CLIP 0.286). Full-frame restyle.
- **Sculpture** = intermediate deviation, CLIP 0.236.
- **Man→woman** = most localized: scene largely preserved (SSIM 0.84, PSNR 23.9) because only the person changes. Its CLIP (0.233) is lowest partly as a **measurement artifact** — CLIP scores the *whole* frame against the instruction, so a local edit leaves most pixels unrelated to "a woman."

## Model size on disk (static Gaussian `.ply`, refined)

The `.ply` stores only the static Gaussians (positions + SH + scale/rot/opacity); the HexPlane deformation network is saved separately and is shared/constant.

| Model | Size |
|---|---|
| Original 4DGS checkpoint (iter 14000) | 23 MB |
| Fauvism (refined) | 27 MB |
| Sculpture (refined) | 24 MB |
| Man→woman (refined) | 25 MB |

## Notes / caveats

- All runs used **default guidance (10.5 / 1.2)**. Per plan: keep defaults; if a result is visually poor, document the config and move on rather than tuning.
- Visual quality of the renders has **not** yet been inspected — metrics alone don't confirm edit fidelity. Renders are under each `point_cloud_refine/<prompt>/iteration_800/`.
- For the upcoming **4D-Scaffold-GS comparison**, hold these axes fixed: same scene, same prompts, same guidance. Compare: peak VRAM (edit_3d / refine_sds), the 5 quality metrics, and on-disk model size. The Scaffold approach is expected to cut disk size and primitive count; watch whether CLIP (edit fidelity) holds, especially with `--freeze_mlp` on vs off.

## Track A: Significance pruning (LightGaussian-style)

`compress_gaussians.py` scores each Gaussian by **opacity × volume** (sigmoid-opacity × product of exp-scales), keeps the top `keep_ratio` fraction, and writes a compacted checkpoint (`.ply` + pruned deformation table/accum; the deformation *network* is global and copied unchanged). The editing pipeline consumes it via new `--deform_path` flags added to `edit_3d.py` / `refine_sds.py` (default = `point_cloud/iteration_14000`, so baseline behavior is unchanged).

**Compaction at keep_ratio=0.5** (`point_cloud_compact50`): 94,427 → 47,214 Gaussians, ply 23.4 → 11.7 MB. Edited under model_path `cook_spinach_compact50`.

### Baseline vs Compact-50% — all 3 prompts

| Prompt | variant | SSIM ↑ | PSNR ↑ | LPIPS-VGG ↓ | LPIPS-Alex ↓ | CLIP ↑ | post-edit pts | refined ply | edit_3d / refine VRAM |
|---|---|---|---|---|---|---|---|---|---|
| Fauvism | baseline | 0.5636 | 12.41 | 0.5696 | 0.4071 | 0.2861 | ~100k | 27 MB | 2.40 / 6.92 |
| Fauvism | compact-50% | 0.5697 | 12.42 | 0.5645 | 0.4020 | 0.2828 | 67k | 16 MB | 1.83 / 6.37 |
| Sculpture | baseline | 0.6892 | 13.21 | 0.4208 | 0.3272 | 0.2364 | ~100k | 24 MB | 2.40 / 6.92 |
| Sculpture | compact-50% | 0.6913 | 13.20 | 0.4160 | 0.3262 | 0.2332 | 53k | 13 MB | 1.63 / 6.16 |
| Man→woman | baseline | 0.8401 | 23.93 | 0.2523 | 0.1157 | 0.2334 | ~100k | 25 MB | 2.40 / 6.92 |
| Man→woman | compact-50% | 0.8403 | 24.05 | 0.2491 | 0.1162 | 0.2335 | 59k | 15 MB | 1.67 / 6.20 |

**Across all three edit types (two global styles + one local semantic edit), compact-50% quality is within noise of baseline** (largest CLIP delta 0.003, PSNR/SSIM/LPIPS all flat or slightly better), while point count drops ~33–47%, refined-model disk ~40%, and edit_3d VRAM ~30%. The favorable compactness↔quality tradeoff holds regardless of edit aggressiveness.

Post-edit point counts differ per prompt because `edit_3d` densifies from the 47,214 compacted seed by different amounts (fauvism 67k > woman 59k > sculpture 53k).

**Next:** sweep `keep_ratio` (0.25, 0.10) to find where quality breaks — the compactness limit.

## Artifacts

- Logs: `baseline_results/logs/{sculpture,man_to_woman,eval_all}.log`, `eval_compact_sculpture.log`, `compact_fauvism_woman.log`
- Baseline refined models: `output/dynerf/cook_spinach/point_cloud_refine/<prompt>/iteration_800/`
- Compact checkpoint: `output/dynerf/cook_spinach/point_cloud_compact50/iteration_14000/`
- Compact edited models: `output/dynerf/cook_spinach_compact50/point_cloud_refine/<prompt>/iteration_800/`
- Videos: `baseline_results/videos/{fauvism,sculpture,man_to_woman}.mp4`
