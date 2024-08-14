from enum import Enum
from typing import Optional, List

from pydantic import BaseModel, Field


class SAMModelType(str, Enum):
    vit_h = "vit_h"
    vit_l = "vit_l"
    vit_b = "vit_b"


class OnnxModelExport(BaseModel):
    source_model_type: SAMModelType = Field(description="Specify the source model type to export to ONNX model format.")
    quantized: Optional[bool] = Field(False, description="Quantize output model.")


class Point(BaseModel):
    x: int
    y: int
    label: int = Field(1, description="1 for positive, 0 for negative, default is positive.")


class SAMPredicate(BaseModel):
    model_type: SAMModelType = Field(SAMModelType.vit_h)
    input_points: List[Point]
