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


metrics = ['MSE', 'RMSE', 'PSNR', 'SSIM', 'LPIPS_Alex', 'LPIPS_VGG']
loss_fn_alex = lpips.LPIPS(net='alex')
loss_fn_vgg = lpips.LPIPS(net='vgg')


def skimage_to_torch(img):
    t = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    tensor = t(skimage.img_as_float(img)).float()
    tensor = tensor[None, 0:3, :, :] * 2 - 1
    return tensor


def compare_images(filename_gt, filename_approx):
    img_gt = skimage.io.imread(filename_gt)
    img_approx = skimage.io.imread(filename_approx)
    mse = skimage.metrics.mean_squared_error(img_gt, img_approx)
    psnr = skimage.metrics.peak_signal_noise_ratio(img_gt, img_approx)
    data_range = img_gt.max() - img_approx.min()
    ssim = skimage.metrics.structural_similarity(
        img_gt, img_approx, data_range=data_range, channel_axis=-1, multichannel=True)

    img0 = skimage_to_torch(img_gt)
    img1 = skimage_to_torch(img_approx)
    d_alex = loss_fn_alex(img0, img1).item()
    d_vgg = loss_fn_vgg(img0, img1).item()

    return {
        'MSE': mse,
        'RMSE': math.sqrt(mse),
        'PSNR': psnr,
        'SSIM': ssim,
        'LPIPS_Alex': d_alex,
        'LPIPS_VGG': d_vgg,
    }
    #print(f'MSE: {mse}')
    #print(f'RMSE: {math.sqrt(mse)}')
    #print(f'PSNR: {psnr}')
    #print(f'SSIM: {ssim}')
    #print(f'LPIPS (Alex): {d_alex}')
    #print(f'LPIPS (VGG): {d_vgg}')
    #print()


def invoke_command(command):
    print(f"Running '{' '.join(command)}'...")
    command_env = os.environ.copy()
    command_env['MKL_SERVICE_FORCE_INTEL'] = '1'
    proc = subprocess.Popen(command, env=command_env)
    proc_status = proc.wait()
    if proc_status != 0:
        raise RuntimeError('subprocess.Popen failed.')


def main():
    matplotlib.rcParams.update({'font.family': 'Linux Biolinum O'})
    matplotlib.rcParams.update({'font.size': 17.5})

    parser = argparse.ArgumentParser(
        prog='vpt_denoise',
        description='Generates volumetric path tracing images.')
    parser.add_argument('--iterations', type=int, default=30000)
    parser.add_argument('--img_idx', default='00000')
    parser.add_argument('--case', default='train')
    parser.add_argument('--gt_scene')
    parser.add_argument('--test_scenes', nargs='+')

    args = parser.parse_args()

    gt_dir = args.gt_scene  # '/mnt/data/3DGS/train/bonsai_default'
    gt_image_dir = os.path.join(gt_dir, args.case, f'ours_{args.iterations}', 'renders')
    if not os.path.exists(gt_image_dir):
        invoke_command(['python3', 'render.py', '-m', gt_dir, '--antialiasing'])
    filename_gt = os.path.join(gt_image_dir, f'{args.img_idx}.png')
    test_dirs = args.test_scenes
    # [
    #     '/mnt/data/3DGS/train/bonsai_ds2',
    #     '/mnt/data/3DGS/train/bonsai_ninasr_b1',
    # ]
    results = []
    for test_dir in test_dirs:
        if not os.path.exists(test_dir):
            raise RuntimeError(f'Error: Directory {test_dir} does not exist.')
        test_name = os.path.basename(test_dir)
        image_dir = os.path.join(test_dir, args.case, f'ours_{args.iterations}', 'renders')
        if not os.path.exists(image_dir):
            invoke_command(['python3', 'render.py', '-m', test_dir, '--antialiasing'])
        print(f"Test '{test_name}'...")
        filename_approx = os.path.join(image_dir, f'{args.img_idx}.png')
        result = compare_images(filename_gt, filename_approx)
        result['name'] = test_name
        results.append(result)

    results = sorted(results, key=lambda d: d['name'])
    print(results)
    #plot_results(base_dir, args.sf, results)


if __name__ == '__main__':
    main()
