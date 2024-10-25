from .upscaler import Upscaler
import torch


class UpscalerPyTorch(Upscaler):
    # For mode see: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
    # mode: 'nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' | 'area' | 'nearest-exact'
    def __init__(self, ss_factor, mode='bicubic', align_corners=None):
        super().__init__(ss_factor)
        self.mode = mode
        self.align_corners = align_corners

    def apply(self, render_width, render_height, upscaled_width, upscaled_height, rendered_image, depth_image):
        return torch.nn.functional.interpolate(
            rendered_image.unsqueeze(0), size=(upscaled_height, upscaled_width),
            mode=self.mode, align_corners=self.align_corners).squeeze(0)

    def query_render_resolution(self, upscaled_width, upscaled_height):
        return super().query_render_resolution(upscaled_width, upscaled_height)
