from .upscaler import Upscaler
import sys
import torch
try:
    import pysrg
    can_use_dlss = True
except ImportError:
    can_use_dlss = False


class UpscalerDLSS(Upscaler):
    def __init__(self, ss_factor):
        super().__init__(ss_factor)
        if ss_factor == 2:
            self.perf_quality_mode = pysrg.DlssPerfQuality.MAX_PERF
        else:
            self.perf_quality_mode = pysrg.DlssPerfQuality.ULTRA_PERFORMANCE
        if ss_factor != 2 and ss_factor != 3:
            print(
                'Error: UpscalerDLSS only supports an upscaling factor of 2 and 3. Falling back to 3.', file=sys.stderr)
        pysrg.set_perf_quality_dlss(self.perf_quality_mode)

    def apply(self, render_width, render_height, upscaled_width, upscaled_height, rendered_image, depth_image):
        exposure_value = 1.0  # Use something else?
        motion_vectors = torch.zeros(
            (rendered_image.shape[1], rendered_image.shape[2], 2), dtype=torch.float16, device=rendered_image.device)
        motion_vectors = torch.permute(motion_vectors, (2, 1, 0))  # H, W, C -> C, H, W (reversed on C++ side)
        upscaled_image = pysrg.apply_supersampling_dlss(
            upscaled_width, upscaled_height, rendered_image, depth_image, motion_vectors, exposure_value)
        pysrg.reset_accumulation()
        return upscaled_image

    def query_render_resolution(self, upscaled_width, upscaled_height):
        render_width, render_height = pysrg.query_optimal_resolution_dlss(
            upscaled_width, upscaled_height)
        return render_width, render_height
