import math, os, sys, torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser

from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from scene import Scene
from gaussian_renderer import GaussianModel, render
from utils.general_utils import safe_state

parser = ArgumentParser(description="Render test views from edited checkpoint")
model    = ModelParams(parser, sentinel=True)
pipeline = PipelineParams(parser)
hyper    = ModelHiddenParams(parser)
parser.add_argument("--iteration", default=-1, type=int)
parser.add_argument("--configs", type=str)
parser.add_argument("--ply_path", type=str)
args = get_combined_args(parser)

ply_path = args.ply_path  # save before merge_hparams may drop it
if args.configs:
    from mmengine import Config
    from utils.params_utils import merge_hparams
    config = Config.fromfile(args.configs)
    args = merge_hparams(args, config)

safe_state(args.quiet if hasattr(args, 'quiet') else False)

dataset   = model.extract(args)
hyperparam = hyper.extract(args)
pipe      = pipeline.extract(args)

gaussians = GaussianModel(dataset.sh_degree, hyperparam)
scene     = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)

print("before edit:", gaussians.get_xyz.shape)
gaussians.load_ply(ply_path)
gaussians.load_model(os.path.join(args.model_path, 'point_cloud', 'iteration_14000'))
print("after edit:", gaussians.get_xyz.shape)

bg_color   = [1,1,1] if dataset.white_background else [0,0,0]
background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

out_base = os.path.join(args.model_path, "test", "ours_800")
renders_dir = Path(out_base) / "renders"
gt_dir      = Path(out_base) / "gt"
renders_dir.mkdir(parents=True, exist_ok=True)
gt_dir.mkdir(parents=True, exist_ok=True)

test_cams = scene.getTestCameras()
print(f"Test cameras: {len(test_cams)}")

to8b = lambda x: (255 * np.clip(x.detach().cpu().numpy(), 0, 1)).astype(np.uint8)

for idx, cam in enumerate(tqdm(test_cams, desc="Rendering test views")):
    pkg = render(cam, gaussians, pipe, background, stage="fine", cam_type=scene.dataset_type)
    img = pkg["render"]
    Image.fromarray(to8b(img).transpose(1,2,0)).save(renders_dir / f"cam_{idx:02d}.png")

    gt = cam.original_image if scene.dataset_type != "PanopticSports" else cam["image"]
    Image.fromarray(to8b(gt).transpose(1,2,0)).save(gt_dir / f"cam_{idx:02d}.png")

print(f"Saved {len(test_cams)} renders → {renders_dir}")
print(f"Saved {len(test_cams)} GT      → {gt_dir}")
