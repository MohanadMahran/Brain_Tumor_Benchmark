"""Model implementations for brain tumor segmentation benchmark.

Provides:
    - UNet3D: 3D U-Net with GroupNorm (~19M params)
    - UNETR: Reduced UNETR with LayerNorm (~19M params)
    - Parameter counting utilities
    - Pre-trained weight adapters
"""

from models.unet3d import UNet3D
from models.unetr import UNETR
from models.param_counter import count_parameters, verify_parameter_parity

__all__ = ["UNet3D", "UNETR", "count_parameters", "verify_parameter_parity"]
