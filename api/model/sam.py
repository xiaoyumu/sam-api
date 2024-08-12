from enum import Enum

from pydantic import BaseModel, Field


class SAMModelType(str, Enum):
    vit_h = "vit_h"
    vit_l = "vit_l"
    vit_b = "vit_b"


class OnnxModelExport(BaseModel):
    source_model_type: SAMModelType = Field(description="Specify the source model type to export to ONNX model format.")
