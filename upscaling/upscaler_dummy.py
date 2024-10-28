import torch
from .upscaler import Upscaler


class UpscalerDummy(Upscaler):
    def __init__(self, ss_factor, **kwargs):
        super().__init__(ss_factor)

    def apply(self, render_width, render_height, upscaled_width, upscaled_height, rendered_image, depth_image):
        return torch.nn.functional.interpolate(
            rendered_image.unsqueeze(0), size=(upscaled_height, upscaled_width),
            mode='bicubic', align_corners=False).squeeze(0)

    def query_render_resolution(self, upscaled_width, upscaled_height):
        return super().query_render_resolution(upscaled_width, upscaled_height)
