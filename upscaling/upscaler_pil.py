import torch

from .upscaler import Upscaler
import numpy as np
from PIL import Image


class UpscalerPIL(Upscaler):
    def __init__(self, ss_factor, resample_algo=Image.Resampling.LANCZOS):
        super().__init__(ss_factor)
        self.resample_algo = resample_algo

    def apply(self, render_width, render_height, upscaled_width, upscaled_height, rendered_image, depth_image):
        numpy_image = rendered_image.cpu().numpy()
        numpy_image = np.clip(numpy_image, 0.0, 1.0)
        numpy_image = np.transpose(numpy_image, (1, 2, 0))
        numpy_image = numpy_image * 255.0
        numpy_image = numpy_image.astype(np.uint8)
        upscaled_image = Image.fromarray(numpy_image).resize(
            size=(upscaled_width, upscaled_height), resample=self.resample_algo)
        upscaled_image = np.transpose(np.array(upscaled_image), (2, 0, 1)).astype(np.float32) / 255.0
        upscaled_image = torch.from_numpy(upscaled_image).to(rendered_image.device)
        return upscaled_image

    def query_render_resolution(self, upscaled_width, upscaled_height):
        return super().query_render_resolution(upscaled_width, upscaled_height)
