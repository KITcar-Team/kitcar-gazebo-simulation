from typing import List

from .base_options import BaseOptions


class TestOptions(BaseOptions):
    dataset_a: List[str] = ["./../../../../data/real_images/maschinen_halle_parking"]
    """path to images of domain A (real images)."""
    dataset_b: List[str] = ["./../../../../data/simulated_images/test_images"]
    """path to images of domain B (simulated images)."""
    results_dir: str = "./results/"
    """saves results here."""
    aspect_ratio: float = 1
    """aspect ratio of result images"""
    is_train: bool = False
    """enable or disable training mode"""


class WassersteinCycleGANTestOptions(TestOptions):
    pass


class CycleGANTestOptions(TestOptions):
    pass
