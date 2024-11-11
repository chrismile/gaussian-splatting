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
import numpy as np
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


def plot_results(base_dir, sf, results):
    for metric in metrics:
        metric_lower = metric.lower()
        fig, ax = plt.subplots()
        fig.set_figwidth(7)
        fig.set_figheight(6)
        ax.set_ylabel(metric)
        #ax.set_title(metric)
        names = []
        values = []
        for result in results:
            result_name = result['name']
            metric_value = result[metric]
            names.append(result_name)
            values.append(metric_value)
        plt.setp(ax.get_xticklabels(), rotation=60, fontsize=14, horizontalalignment='right')
        ax.bar(names, values)
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, f'x{sf}_{metric_lower}.pdf'), bbox_inches='tight', pad_inches=0.01)


def plot_timings(base_dir, sf, results):
    fig, ax = plt.subplots()
    fig.set_figwidth(7)
    fig.set_figheight(6)
    ax.set_ylabel('time (s)')
    # ax.set_title(metric)
    names = []
    values = []
    for result in results:
        result_name = result['name']
        time_value = result['time']
        names.append(result_name)
        values.append(time_value)
    plt.setp(ax.get_xticklabels(), rotation=60, fontsize=14, horizontalalignment='right')
    ax.bar(names, values)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, f'x{sf}_time.pdf'), bbox_inches='tight', pad_inches=0.01)


def main():
    matplotlib.rcParams.update({'font.family': 'Linux Biolinum O'})
    matplotlib.rcParams.update({'font.size': 17.5})

    parser = argparse.ArgumentParser(
        prog='vpt_denoise',
        description='Generates volumetric path tracing images.')
    parser.add_argument('-m', '--dir')
    parser.add_argument('--iterations', type=int, default=30000)
    parser.add_argument('-s', '--sf', type=int, default=2)
    #parser.add_argument('--img_idx', default='00000')
    parser.add_argument('--img_idx_min', type=int, default=0)
    parser.add_argument('--img_idx_max', type=int, default=9)
    parser.add_argument('--case', default='train')
    args = parser.parse_args()

    if os.path.exists('/mnt/data/3DGS'):
        base_dir = '/mnt/data/3DGS/train/bonsai_default/train/'
    elif os.path.exists('/home/neuhauser/datasets/3dgs/nerf/3DGS'):
        base_dir = '/home/neuhauser/datasets/3dgs/nerf/3DGS/train/bonsai_default/train/'
    if args.dir is not None:
        base_dir = os.path.join(args.dir, args.case)
    base_path = f'ours_{args.iterations}_x{args.sf}_'
    num_images = args.img_idx_max - args.img_idx_min + 1
    results = []
    upscale_timings = []
    for test_dir in os.listdir(base_dir):
        if not test_dir.startswith(base_path):
            continue
        test_name = test_dir[len(base_path):]
        if 'opencv_EDSR' in test_name:
            continue
        print(f"Test '{test_name}'...")
        result = {'name': test_name}
        for metric in metrics:
            result[metric] = 0.0
        for img_idx in range(args.img_idx_min, args.img_idx_max + 1):
            img_idx_string = f'{img_idx:05d}'
            filename_gt = os.path.join(base_dir, test_dir, 'renders', f'{img_idx_string}.png')
            filename_approx = os.path.join(base_dir, test_dir, 'upscaled', f'{img_idx_string}.png')
            result_frame = compare_images(filename_gt, filename_approx)
            for metric in metrics:
                result[metric] += result_frame[metric] / num_images
        results.append(result)

        # Get timings.
        timings_path = os.path.join(base_dir, test_dir, 'times_upscale.txt')
        with open(timings_path) as file:
            upscale_timings_current = [float(line.rstrip()) for line in file]
            upscale_timings_current = np.array(upscale_timings_current)
            upscale_timings.append({'name': test_name, 'time': np.mean(upscale_timings_current)})

    results = sorted(results, key=lambda d: d['name'])
    upscale_timings = sorted(upscale_timings, key=lambda d: d['name'])
    plot_results(base_dir, args.sf, results)
    plot_timings(base_dir, args.sf, upscale_timings)


if __name__ == '__main__':
    main()
