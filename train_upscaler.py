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
from upscaling.espcn import *
from upscaling.upscaler_model import UpscalerModel


def run_optimizer(dataset : ModelParams, num_batches : int, pipeline : PipelineParams):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=-1, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        views = scene.getTestCameras()
        train_test_exp = dataset.train_test_exp
        out_path = os.path.join(dataset.model_path, "upscaling")
        makedirs(out_path, exist_ok=True)
        sf = 2

    upscaler_model = ESPCN(in_channels=4, out_channels=3, channels=64, upscale_factor=sf)
    upscaler_model.to(device=torch.device("cuda"))
    def init_weights(m):
        if isinstance(m, torch.nn.Conv2d):
            m.weight.data.normal_(0.0, 0.02)
            m.bias.data.normal_(0.0, 0.02)
        elif hasattr(m, "weight"):
            m.weight.data.normal_(0.0, 0.02)
            # m.weight.data.fill_(0.01)
    upscaler_model.apply(init_weights)
    upscaler_model.train()
    upscaler = UpscalerModel(ss_factor=sf, model=upscaler_model, train_model=True)

    loss_function = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(upscaler_model.parameters(), lr=1e-4)

    print('Starting optimization...')
    running_loss = 0.0
    num_its_per_batch = len(views)
    for batch_idx in tqdm(range(num_batches), desc="Batch"):
        for idx, view in enumerate(views):
            optimizer.zero_grad()

            with torch.no_grad():
                render_out = render(
                    view, gaussians, pipeline, background, use_trained_exp=train_test_exp)
                render_out_ss = render(
                    view, gaussians, pipeline, background, use_trained_exp=train_test_exp, upscaler=upscaler)
            rendering = render_out["render"]
            rendering_upscaled = render_out_ss["render"]

            if rendering_upscaled.shape != rendering.shape:
                rendering = rendering[:, :rendering_upscaled.shape[1], :rendering_upscaled.shape[2]]
            loss = loss_function(rendering_upscaled, rendering)
            #loss.register_hook(lambda grad: print(grad))
            #rendering_upscaled.register_hook(lambda grad: print(grad))
            loss.backward()
            optimizer.step()
            #print(rendering_upscaled.grad)
            #print(upscaling_model.feature_maps[0].weight.grad)
            running_loss += loss.item()
        train_loss_avg = running_loss / num_its_per_batch
        print(f'Train loss: {train_loss_avg}')
        running_loss = 0.0

    upscaler_model.eval()
    torch.save(upscaler_model, os.path.join(out_path, "espcn_{}.pt".format(num_batches)))
    print('Quitting program...')


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--num_batches", default=1024, type=int)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    run_optimizer(model.extract(args), args.num_batches, pipeline.extract(args))
