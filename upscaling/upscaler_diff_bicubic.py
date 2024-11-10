from .upscaler import Upscaler
import torch
from diff_bicubic import BicubicInterpolation


class UpscalerDiffBicubic(Upscaler):
    # For mode see: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
    # mode: 'nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' | 'area' | 'nearest-exact'
    def __init__(self, ss_factor):
        super().__init__(ss_factor)

    def apply(
            self, render_width, render_height, upscaled_width, upscaled_height,
            rendered_image, depth_image, gradient_image):
        num_channels = gradient_image.shape[0] // 3
        gradient_image = gradient_image.reshape((num_channels, 3, gradient_image.shape[1], gradient_image.shape[2]))
        gradient_image[:, 0:2, :, :] = -gradient_image[:, 0:2, :, :]

        # For comparing analytic gradients with numerical ones.
        #dy, dx = torch.gradient(rendered_image, dim=[1, 2])
        #dxy = torch.gradient(dx, dim=1)
        #for i in range(3):
        #    gradient_image[i, 0, :, :] = dx[i, :, :]
        #    gradient_image[i, 1, :, :] = dy[i, :, :]
        #    gradient_image[i, 2, :, :] = dxy[0][i, :, :]

        upscaled_image = BicubicInterpolation.apply(
            rendered_image, gradient_image, upscaled_width, upscaled_height)
        return upscaled_image


    def query_render_resolution(self, upscaled_width, upscaled_height):
        return super().query_render_resolution(upscaled_width, upscaled_height)
