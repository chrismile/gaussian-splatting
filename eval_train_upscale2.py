# BSD 2-Clause License
#
# Copyright (c) 2024, Christoph Neuhauser
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import sys
import math
import argparse
import subprocess
import skimage
import skimage.io
import skimage.metrics
import lpips
import torch
import torchvision
import matplotlib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from scene import Scene
from gaussian_renderer import render, GaussianModel
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.general_utils import safe_state


metrics = ['MSE', 'RMSE', 'PSNR', 'SSIM', 'LPIPS_Alex', 'LPIPS_VGG']
loss_fn_alex = lpips.LPIPS(net='alex')
loss_fn_vgg = lpips.LPIPS(net='vgg')


def skimage_to_torch(img):
    t = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    tensor = t(skimage.img_as_float(img)).float()
    tensor = tensor[None, 0:3, :, :] * 2 - 1
    return tensor


def compare_images(tensor_gt, tensor_approx):
    tensor_gt = torch.clip(tensor_gt, 0.0, 1.0)
    tensor_approx = torch.clip(tensor_approx, 0.0, 1.0)
    torchvision.utils.save_image(tensor_gt, os.path.join('test_gt.png'))
    torchvision.utils.save_image(tensor_approx, os.path.join('test_approx.png'))

    img_gt = tensor_gt.cpu().numpy().transpose((1, 2, 0))
    img_approx = tensor_approx.cpu().numpy().transpose((1, 2, 0))
    mse = skimage.metrics.mean_squared_error(img_gt, img_approx)
    psnr = skimage.metrics.peak_signal_noise_ratio(img_gt, img_approx)
    data_range = img_gt.max() - img_approx.min()
    ssim = skimage.metrics.structural_similarity(
        img_gt, img_approx, data_range=data_range, channel_axis=-1, multichannel=True)

    #img0 = tensor_gt.cuda()  # skimage_to_torch(img_gt)
    #img1 = tensor_approx.cuda()  # skimage_to_torch(img_approx)
    #d_alex = loss_fn_alex(img0, img1).item()
    #d_vgg = loss_fn_vgg(img0, img1).item()
    d_alex = 0.0
    d_vgg = 0.0

    return {
        'MSE': mse,
        'RMSE': math.sqrt(mse),
        'PSNR': psnr,
        'SSIM': ssim,
        'LPIPS_Alex': d_alex,
        'LPIPS_VGG': d_vgg,
    }


def main():
    matplotlib.rcParams.update({'font.family': 'Linux Biolinum O'})
    matplotlib.rcParams.update({'font.size': 17.5})

    parser = argparse.ArgumentParser(
        description='Generates volumetric path tracing images.')
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    pipeline = pipeline.extract(args)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    result = {}
    for metric in metrics:
        result[metric] = 0.0

    timings_file_path = os.path.join(args.model_path, 'train_time.txt')
    if os.path.exists(timings_file_path):
        with open(timings_file_path) as timings_file:
            print(f'{float(timings_file.read())}s')

    dataset = model.extract(args)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    views = scene.getTestCameras()
    #views = scene.getTrainCameras()
    #num_images = 10
    num_images = len(views)
    for idx, view in enumerate(views):
        gt = view.original_image[0:3, :, :]
        render_out = render(
            view, gaussians, pipeline, background, use_trained_exp=dataset.train_test_exp)

        result_frame = compare_images(gt, render_out["render"][0:3, :, :])
        for metric in metrics:
            result[metric] += result_frame[metric] / num_images

        #if idx >= num_images:
        #    break

    print(result)
    #plot_results(base_dir, args.sf, results)


if __name__ == '__main__':
    with torch.no_grad():
        main()
