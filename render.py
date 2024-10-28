#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from upscaling.upscaler_dlss import UpscalerDLSS
from upscaling.upscaler_pytorch import UpscalerPyTorch
from upscaling.upscaler_model import UpscalerModel
from upscaling.upscaler_cv2 import UpscalerOpenCV
from upscaling.upscaler_pil import pil_resample_algo_from_name, UpscalerPIL
try:
    from upscaling.upscaler_torchsr import UpscalerTorchSR
    torchsr_found = True
except ImportError:
    from upscaling.upscaler_dummy import UpscalerDummy as UpscalerTorchSR
    torchsr_found = False


def render_set(
        model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp,
        sf: int, upscaling_method: str, upscaling_param: str):
    dir_name = f"ours_{iteration}"
    if sf is not None and sf != 1:
        dir_name += f"_x{sf}"
    if upscaling_method is not None:
        dir_name += f"_{upscaling_method}"
    if upscaling_param is not None:
        dir_name += f"_{upscaling_param}"
    render_path = os.path.join(model_path, name, dir_name, "renders")
    small_path = os.path.join(model_path, name, dir_name, "small")
    upscaled_path = os.path.join(model_path, name, dir_name, "upscaled")
    gts_path = os.path.join(model_path, name, dir_name, "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(small_path, exist_ok=True)
    makedirs(upscaled_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    upscaler = None
    upscaling_method_lower = upscaling_method.lower() if upscaling_method is not None else None
    upscaling_param_lower = upscaling_param.lower() if upscaling_param is not None else None
    if upscaling_method_lower == 'dlss':
        upscaler = UpscalerDLSS(ss_factor=sf)
    elif upscaling_method_lower == 'pytorch':
        upscaler = UpscalerPyTorch(ss_factor=sf, mode=upscaling_param_lower)
    elif upscaling_method_lower == 'opencv':
        algo_name = upscaling_param
        # algo_name = 'EDSR'
        # algo_name = 'ESPCN'
        # algo_name = 'FSRCNN'
        # algo_name = 'LapSRN'
        upscaler = UpscalerOpenCV(
            ss_factor=sf, algo_name=algo_name, model_path=f'/mnt/data/DL/img_upscale_models/x{sf}/{algo_name}_x{sf}.pb')
    elif upscaling_method_lower == 'pil':
        algo = pil_resample_algo_from_name(upscaling_param_lower)
        upscaler = UpscalerPIL(ss_factor=sf, resample_algo=algo)
    elif upscaling_method_lower == 'model':
        upscaler_model = torch.load(os.path.join(model_path, "upscaling", upscaling_param_lower))
        upscaler_model.eval()
        upscaler = UpscalerModel(ss_factor=sf, model=upscaler_model)
    elif upscaling_method_lower == 'torchsr' and torchsr_found:
        upscaler = UpscalerTorchSR(ss_factor=sf, model_name=upscaling_param_lower)

    round_sizes = 1
    if upscaler is not None and not upscaler.get_supports_fractional():
        round_sizes = upscaler.get_ss_factor()

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_out = render(
            view, gaussians, pipeline, background, use_trained_exp=train_test_exp, round_sizes=round_sizes)
        rendering = render_out["render"][0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        if sf != 1 and upscaler is not None:
            render_out_ss = render(
                view, gaussians, pipeline, background, use_trained_exp=train_test_exp, upscaler=upscaler,
                round_sizes=round_sizes)
            if "render_small" in render_out_ss:
                rendering_small = render_out_ss["render_small"][0:3, :, :]
                torchvision.utils.save_image(rendering_small, os.path.join(small_path, '{0:05d}'.format(idx) + ".png"))
            rendering_upscaled = render_out_ss["render"][0:3, :, :]
            torchvision.utils.save_image(rendering_upscaled, os.path.join(upscaled_path, '{0:05d}'.format(idx) + ".png"))
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        if idx >= 10:
            break


def render_sets(
        dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool,
        sf: int, upscaling_method: str, upscaling_param: str):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(
                 dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline,
                 background, dataset.train_test_exp, sf, upscaling_method, upscaling_param)

        if not skip_test:
             render_set(
                 dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline,
                 background, dataset.train_test_exp, sf, upscaling_method, upscaling_param)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--sf", default=1, type=int)
    parser.add_argument("--upscaling_method", default=None)
    parser.add_argument("--upscaling_param", default=None)
    args = get_combined_args(parser)
    upscaling_method = None
    upscaling_param = None
    if hasattr(args, 'upscaling_method'):
        upscaling_method = args.upscaling_method
    if hasattr(args, 'upscaling_param'):
        upscaling_param = args.upscaling_param
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(
        model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test,
        args.sf, upscaling_method, upscaling_param)
