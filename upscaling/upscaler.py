from enum import Enum
import abc


class UpscalerType(Enum):
    NONE = 0
    DLSS = 1
    PYTORCH = 2
    CUSTOM = 3


class Upscaler(abc.ABC):
    def __init__(self, ss_factor):
        super().__init__()
        self.ss_factor = ss_factor

    def get_ss_factor(self) -> int:
        return self.ss_factor

    @abc.abstractmethod
    def apply(
            self, render_width: int, render_height: int, upscaled_width: int, upscaled_height: int,
            rendered_image, depth_image, gradient_image):
        raise NotImplementedError()

    def query_render_resolution(self, upscaled_width: int, upscaled_height: int) -> tuple[int, int]:
        return int(upscaled_width) // self.ss_factor, int(upscaled_height) // self.ss_factor

    def get_supports_fractional(self) -> bool:
        return False  # Set to False as a test
