import torch

from .upscaler import Upscaler
import numpy as np
import cv2


class UpscalerOpenCV(Upscaler):
    # https://docs.opencv.org/4.x/d8/d11/classcv_1_1dnn__superres_1_1DnnSuperResImpl.html#ab4d5e45240e7dbc436f077d34bff8854
    # algo: 'edsr' | 'espcn' | 'fsrcnn' | 'lapsrn'
    def __init__(self, ss_factor, algo_name, model_path=None):
        super().__init__(ss_factor)
        sr_model = cv2.dnn_superres.DnnSuperResImpl_create()
        if model_path is not None:
            sr_model.readModel(model_path)
        sr_model.setModel(algo_name.lower(), ss_factor)
        self.sr_model = sr_model

    def apply(self, render_width, render_height, upscaled_width, upscaled_height, rendered_image, depth_image):
        numpy_image = rendered_image.cpu().numpy()
        numpy_image = np.clip(numpy_image, 0.0, 1.0)
        numpy_image = np.transpose(numpy_image, (1, 2, 0))
        cv2_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR) * 255.0
        upscaled_image = self.sr_model.upsample(cv2_image)
        upscaled_image = cv2.cvtColor(upscaled_image, cv2.COLOR_BGR2RGB)
        upscaled_image = np.transpose(upscaled_image, (2, 0, 1)).astype(np.float32) / 255.0
        upscaled_image = torch.from_numpy(upscaled_image).to(rendered_image.device)
        return upscaled_image

    def query_render_resolution(self, upscaled_width, upscaled_height):
        return super().query_render_resolution(upscaled_width, upscaled_height)

    def get_supports_fractional(self) -> bool:
        return False
