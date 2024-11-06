from .upscaler import Upscaler
import torch


class UpscalerModel(Upscaler):
    def __init__(self, ss_factor, model, train_model=False):
        super().__init__(ss_factor)
        self.model = model
        self.train_model = train_model

    def apply(
            self, render_width, render_height, upscaled_width, upscaled_height,
            rendered_image, depth_image, gradient_image):
        rendered_image_orig = rendered_image
        input_image = torch.cat([rendered_image_orig, depth_image], dim=0).unsqueeze(0)
        with torch.set_grad_enabled(self.train_model):
            rendered_image = self.model(input_image).squeeze(0)
            rendered_image = rendered_image.clamp(0, 1)
        return rendered_image

    def query_render_resolution(self, upscaled_width, upscaled_height):
        return super().query_render_resolution(upscaled_width, upscaled_height)

    def get_supports_fractional(self) -> bool:
        return False
