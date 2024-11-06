import torch
from torchsr.models import *
from .upscaler import Upscaler


class UpscalerTorchSR(Upscaler):
    def __init__(self, ss_factor, model_name='edsr'):
        super().__init__(ss_factor)
        if model_name == 'edsr':
            self.model = edsr(scale=ss_factor, pretrained=True)
        elif model_name == 'vdsr':
            self.model = vdsr(scale=ss_factor, pretrained=True)
        elif model_name == 'rdn':
            self.model = rdn(scale=ss_factor, pretrained=True)
        elif model_name == 'rcan':
            self.model = rcan(scale=ss_factor, pretrained=True)
        elif model_name == 'ninasr_b0':
            self.model = ninasr_b0(scale=ss_factor, pretrained=True)
        elif model_name == 'ninasr_b1':
            self.model = ninasr_b1(scale=ss_factor, pretrained=True)
        elif model_name == 'ninasr_b2':
            self.model = ninasr_b2(scale=ss_factor, pretrained=True)
        elif model_name == 'carn':
            self.model = carn(scale=ss_factor, pretrained=True)
        else:
            raise RuntimeError(f'Error: Unsupported super resolution model name \'{model_name}\'.')
        self.model = self.model.to(torch.device('cuda'))

    def apply(
            self, render_width, render_height, upscaled_width, upscaled_height,
            rendered_image, depth_image, gradient_image):
        input_image = rendered_image.clamp(0, 1).unsqueeze(0)
        rendered_image = self.model(input_image).squeeze(0)
        rendered_image = rendered_image.clamp(0, 1)
        return rendered_image

    def query_render_resolution(self, upscaled_width, upscaled_height):
        return super().query_render_resolution(upscaled_width, upscaled_height)

    def get_supports_fractional(self) -> bool:
        return False
