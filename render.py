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
from upscaling.upscaler_pil import UpscalerPIL


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    small_path = os.path.join(model_path, name, "ours_{}".format(iteration), "small")
    upscaled_path = os.path.join(model_path, name, "ours_{}".format(iteration), "upscaled")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(small_path, exist_ok=True)
    makedirs(upscaled_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    sf = 2

    #upscaler = None
    upscaler = UpscalerDLSS(ss_factor=sf)
    #upscaler = UpscalerPyTorch(ss_factor=sf)

    #algo_name = 'EDSR'
    #algo_name = 'ESPCN'
    #algo_name = 'FSRCNN'
    #algo_name = 'LapSRN'
    #upscaler = UpscalerOpenCV(
    #    ss_factor=sf, algo_name=algo_name, model_path=f'/mnt/data/DL/img_upscale_models/x{sf}/{algo_name}_x{sf}.pb')

    #upscaler = UpscalerPIL(ss_factor=sf)

    #upscaler_model = torch.load(os.path.join(model_path, "upscaling", "espcn_1024.pt"))
    #upscaler_model.eval()
    #upscaler = UpscalerModel(ss_factor=sf, model=upscaler_model)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_out = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp)
        rendering = render_out["render"]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        if sf != 1 and upscaler is not None:
            render_out_ss = render(
                view, gaussians, pipeline, background, use_trained_exp=train_test_exp, upscaler=upscaler)
            if "render_small" in render_out_ss:
                rendering_small = render_out_ss["render_small"]
                torchvision.utils.save_image(rendering_small, os.path.join(small_path, '{0:05d}'.format(idx) + ".png"))
            rendering_upscaled = render_out_ss["render"]
            torchvision.utils.save_image(rendering_upscaled, os.path.join(upscaled_path, '{0:05d}'.format(idx) + ".png"))
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)