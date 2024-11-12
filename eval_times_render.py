import os
import numpy as np


def load_timings_array(timings_path):
    with open(timings_path) as file:
        upscale_timings_current = [float(line.rstrip()) for line in file]
        upscale_timings_current = np.array(upscale_timings_current)
        return upscale_timings_current


def main():
    if os.path.exists('/mnt/data/3DGS'):
        base_dir = '/mnt/data/3DGS/train/garden_default/train/'
    elif os.path.exists('/home/neuhauser/datasets/3dgs/nerf/3DGS'):
        base_dir = '/home/neuhauser/datasets/3dgs/nerf/3DGS/train/garden_default/train/'
    timings_upscale = load_timings_array(os.path.join(base_dir, 'ours_30000_x3_diff_bicubic', 'times_upscale.txt'))
    timings_with_grad = load_timings_array(os.path.join(base_dir, 'ours_30000_x3_diff_bicubic', 'times_render_small.txt'))
    timings_no_grad = load_timings_array(os.path.join(base_dir, 'ours_30000_x3_pytorch_bicubic', 'times_render_small.txt'))
    timings_full_res = load_timings_array(os.path.join(base_dir, 'ours_30000_x3_pytorch_bicubic', 'times_render.txt'))
    diff_time = timings_with_grad.mean() - timings_no_grad.mean()
    print(f'Time upscale diff-bicubic: {timings_upscale.mean() * 1e3}ms')
    print(f'Time render with grad: {timings_with_grad.mean() * 1e3}ms')
    print(f'Time render no grad: {timings_no_grad.mean() * 1e3}ms')
    print(f'Time render full res: {timings_full_res.mean() * 1e3}ms')
    print(f'Difference render: {diff_time * 1e3}ms')
    print()
    for sf in [2, 3, 4]:
        print(f'x{sf}:')
        timings_hires_render = load_timings_array(os.path.join(base_dir, f'ours_30000_x{sf}_pytorch_bicubic', 'times_hires_render.txt'))
        timings_lowres_render_grad = load_timings_array(os.path.join(base_dir, f'ours_30000_x{sf}_diff_bicubic', 'times_hires_render_small.txt'))
        timings_lowres_render_nograd = load_timings_array(os.path.join(base_dir, f'ours_30000_x{sf}_pytorch_bicubic', 'times_hires_render_small.txt'))
        timings_upscale_diff_bicubic = load_timings_array(os.path.join(base_dir, f'ours_30000_x{sf}_diff_bicubic', 'times_hires_upscale.txt'))
        timings_upscale_pytorch_bicubic = load_timings_array(os.path.join(base_dir, f'ours_30000_x{sf}_pytorch_bicubic', 'times_hires_upscale.txt'))
        timings_upscale_ninasr_b1 = load_timings_array(os.path.join(base_dir, f'ours_30000_x{sf}_torchsr_NinaSR_b1', 'times_hires_upscale.txt'))
        #timings_upscale_edsr = load_timings_array(os.path.join(base_dir, f'ours_30000_x{sf}_torchsr_edsr', 'times_hires_upscale.txt'))
        print(f'Time hires render: {timings_hires_render.mean() * 1e3}ms')
        print(f'Time lowres render grad: {timings_lowres_render_grad.mean() * 1e3}ms')
        print(f'Time lowres render nograd: {timings_lowres_render_nograd.mean() * 1e3}ms')
        print(f'Time upscale diff-bicubic: {timings_upscale_diff_bicubic.mean() * 1e3}ms')
        print(f'Time upscale PyTorch bicubic: {timings_upscale_pytorch_bicubic.mean() * 1e3}ms')
        print(f'Time upscale NinaSR-B1: {timings_upscale_ninasr_b1.mean() * 1e3}ms')
        render_hires = timings_hires_render.mean()
        total_time_diff_bicubic = timings_upscale_diff_bicubic.mean() + timings_lowres_render_grad.mean()
        total_time_pytorch_bicubic = timings_upscale_pytorch_bicubic.mean() + timings_lowres_render_nograd.mean()
        total_time_ninasr_b1 = timings_upscale_ninasr_b1.mean() + timings_lowres_render_nograd.mean()
        print(f'Speedup diff-bicubic: {render_hires / total_time_diff_bicubic}')
        print(f'Speedup PyTorch bicubic: {render_hires / total_time_pytorch_bicubic}')
        print(f'Speedup NinaSR-B1: {render_hires / total_time_ninasr_b1}')
        #print(f'Time upscale EDSR: {timings_upscale_edsr.mean() * 1e3}ms')
        print()


if __name__ == '__main__':
    main()
