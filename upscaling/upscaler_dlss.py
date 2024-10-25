from .upscaler import Upscaler
import torch
try:
    import pysrg
    can_use_dlss = True
except ImportError:
    can_use_dlss = False


class UpscalerDLSS(Upscaler):
    def __init__(self, ss_factor):
        super().__init__(ss_factor)

    def apply(self, render_width, render_height, upscaled_width, upscaled_height, rendered_image, depth_image):
        exposure_value = 1.0  # Use something else?
        motion_vectors = torch.zeros(
            (rendered_image.shape[1], rendered_image.shape[2], 2), dtype=torch.float16, device=rendered_image.device)
        motion_vectors = torch.permute(motion_vectors, (2, 1, 0))  # H, W, C -> C, H, W (reversed on C++ side)
        return pysrg.apply_supersampling_dlss(
            upscaled_width, upscaled_height, rendered_image, depth_image, motion_vectors, exposure_value)

    def query_render_resolution(self, upscaled_width, upscaled_height):
        pysrg.set_perf_quality_dlss(pysrg.DlssPerfQuality.MAX_PERF)
        render_width, render_height = pysrg.query_optimal_resolution_dlss(
            upscaled_width, upscaled_height)
        return render_width, render_height
