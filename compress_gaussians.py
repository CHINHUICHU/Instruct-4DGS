"""LightGaussian-style significance pruning of a trained 4DGS checkpoint.

Scores each Gaussian by (opacity * volume), keeps the top `keep_ratio` fraction,
and writes a compacted checkpoint (point_cloud.ply + copied/pruned deformation
files) that the editing pipeline can consume via --ply_path / load_model.
The deformation network is global, so pruning points does not require retraining it.
"""
import os
import torch
import numpy as np
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, ModelHiddenParams, get_combined_args
from gaussian_renderer import GaussianModel


def main():
    parser = ArgumentParser(description="Compact a trained 4DGS checkpoint by significance pruning.")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--configs", type=str, required=True)
    parser.add_argument("--ply_path", type=str, required=True, help="Trained point_cloud.ply to compact.")
    parser.add_argument("--deform_path", type=str, required=True,
                        help="Dir holding deformation.pth/deformation_table.pth/deformation_accum.pth for --ply_path.")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to write the compacted checkpoint.")
    parser.add_argument("--keep_ratio", type=float, default=0.5, help="Fraction of Gaussians to keep (top by score).")
    args = get_combined_args(parser)

    if args.configs:
        from mmengine import Config
        from utils.params_utils import merge_hparams
        config = Config.fromfile(args.configs)
        args = merge_hparams(args, config)

    dataset = model.extract(args)
    hyper = hyperparam.extract(args)

    gaussians = GaussianModel(dataset.sh_degree, hyper)
    gaussians.load_ply(args.ply_path)
    gaussians.load_model(args.deform_path)

    n0 = gaussians._xyz.shape[0]

    with torch.no_grad():
        opacity = gaussians.get_opacity.squeeze(-1)          # [N], sigmoid-activated
        volume = gaussians.get_scaling.prod(dim=1)            # [N], product of exp-scales
        score = opacity * volume

    keep = max(1, int(round(n0 * args.keep_ratio)))
    keep_idx = torch.argsort(score, descending=True)[:keep]
    mask = torch.zeros(n0, dtype=torch.bool, device=score.device)
    mask[keep_idx] = True

    gaussians._xyz = gaussians._xyz[mask]
    gaussians._features_dc = gaussians._features_dc[mask]
    gaussians._features_rest = gaussians._features_rest[mask]
    gaussians._opacity = gaussians._opacity[mask]
    gaussians._scaling = gaussians._scaling[mask]
    gaussians._rotation = gaussians._rotation[mask]
    gaussians._deformation_table = gaussians._deformation_table[mask]
    gaussians._deformation_accum = gaussians._deformation_accum[mask]

    os.makedirs(args.output_dir, exist_ok=True)
    ply_out = os.path.join(args.output_dir, "point_cloud.ply")
    gaussians.save_ply(ply_out)
    gaussians.save_deformation(args.output_dir)

    n1 = gaussians._xyz.shape[0]
    ply_mb = os.path.getsize(ply_out) / 1e6
    print(f"[COMPACT] keep_ratio={args.keep_ratio} kept {n1}/{n0} ({100.0*n1/n0:.1f}%) "
          f"-> {ply_out} ({ply_mb:.1f} MB)")


if __name__ == "__main__":
    main()
