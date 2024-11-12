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


if __name__ == '__main__':
    main()
