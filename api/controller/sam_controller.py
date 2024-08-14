import io
import os.path
import tempfile
import warnings
from logging import Logger
from pathlib import Path
from typing import Dict, List

import arrow
import cv2
import numpy as np
import torch
import torchvision
from onnxruntime.quantization import QuantType  # type: ignore
from onnxruntime.quantization.quantize import quantize_dynamic  # type: ignore

from api.config.settings import Settings
from api.model.sam import SAMModelType, Point
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from segment_anything.modeling import Sam
from segment_anything.utils.onnx import SamOnnxModel


class SAMController(object):
    def __init__(self, settings: Settings, logger: Logger):
        self.settings = settings
        self.logger = logger
        self.checkpoint_dir = "E:\\ai\\models\\sam"
        self.model_types = {
            SAMModelType.vit_h: "sam_vit_h_4b8939.pth",
            SAMModelType.vit_l: "sam_vit_l_0b3195.pth",
            SAMModelType.vit_b: "sam_vit_b_01ec64.pth",
        }
        default_model_type = SAMModelType.vit_h
        print("Environment checking ...")
        print("PyTorch version:", torch.__version__)
        print("Torchvision version:", torchvision.__version__)
        self.cuda_is_available = torch.cuda.is_available()
        print("CUDA is available:", self.cuda_is_available)
        self.device_type = "cpu"
        self.device_type = "cuda" if self.cuda_is_available else "cpu"

        # checkpoint = os.path.join(self.checkpoint_dir, self.model_types[default_model_type])
        #
        # print(f"Loading SAM model {checkpoint} in {self.device_type} mode ...")
        #
        # sam = sam_model_registry[default_model_type.value](checkpoint=checkpoint)
        # sam.to(device=self.device_type)
        # self.predictor = SamPredictor(sam)

        self._sam_models: Dict[SAMModelType: Sam] = {}

    def _get_sam_model(self, model_type: SAMModelType) -> Sam:
        if model_type not in self._sam_models:
            checkpoint = os.path.join(self.checkpoint_dir, self.model_types[model_type])
            print(f"Loading SAM model {checkpoint} in {self.device_type} mode ...")

            sam = sam_model_registry[model_type.value](checkpoint=checkpoint)
            sam.to(device=self.device_type)
            self._sam_models[model_type] = sam
        return self._sam_models[model_type]

    def _get_predictor(self, model_type: SAMModelType) -> SamPredictor:
        return SamPredictor(self._get_sam_model(model_type))

    async def export_onnx_model(self, model_type: SAMModelType, quantize: bool = False):
        checkpoint = os.path.join(self.checkpoint_dir, self.model_types[model_type])
        sam = sam_model_registry[model_type.value](checkpoint=checkpoint)
        onnx_model = SamOnnxModel(sam, return_single_mask=True,
                                  use_stability_score=False,
                                  return_extra_metrics=False)

        dynamic_axes = {
            "point_coords": {1: "num_points"},
            "point_labels": {1: "num_points"},
        }

        embed_dim = sam.prompt_encoder.embed_dim
        embed_size = sam.prompt_encoder.image_embedding_size
        mask_input_size = [4 * x for x in embed_size]
        dummy_inputs = {
            "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
            "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float),
            "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),
            "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
            "has_mask_input": torch.tensor([1], dtype=torch.float),
            "orig_im_size": torch.tensor([1500, 2250], dtype=torch.float),
        }
        output_names = ["masks", "iou_predictions", "low_res_masks"]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            with io.BytesIO() as exported:
                torch.onnx.export(
                    onnx_model,
                    tuple(dummy_inputs.values()),
                    exported,
                    export_params=True,
                    verbose=False,
                    opset_version=17,
                    do_constant_folding=True,
                    input_names=list(dummy_inputs.keys()),
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                )
                if not quantize:
                    return exported.getvalue()
                files_to_cleanup = []
                with tempfile.NamedTemporaryFile(prefix=f"onnx_model_{model_type.value}", suffix=".onnx", mode="wb",
                                                 delete=False) as model_file:
                    model_file.write(exported.getvalue())
                    model_file.flush()
                    print(f"ONNX Model exported to {model_file.name} .")
                    files_to_cleanup.append(model_file.name)

                try:
                    time_stamp = int(arrow.Arrow.now().float_timestamp * 1000)
                    with tempfile.NamedTemporaryFile(prefix=f"onnx_model_{model_type.value}_quantized_",
                                                     suffix=".onnx",
                                                     mode="wb",
                                                     delete=False) as quantized_model_file:
                        files_to_cleanup.append(quantized_model_file.name)
                    quantize_dynamic(
                        model_input=Path(model_file.name),
                        model_output=Path(quantized_model_file.name),
                        optimize_model=True,
                        per_channel=False,
                        reduce_range=False,
                        weight_type=QuantType.QUInt8,
                    )
                    with open(quantized_model_file.name, "rb") as output_file:
                        return output_file.read()
                finally:
                    for f in files_to_cleanup:
                        if os.path.exists(f):
                            os.remove(f)

    async def get_image_embeddings(self, image_bytes: bytes, model_type = SAMModelType.vit_h) -> np.ndarray:
        buffer = np.asarray(bytearray(image_bytes), dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        predictor = self._get_predictor(model_type)
        predictor.set_image(image)
        image_embedding = predictor.get_image_embedding().cpu().numpy()
        # np.save("embedding_file.npy", image_embedding)
        return image_embedding

    async def get_image_embeddings_bytes(self, image_bytes: bytes, model_type = SAMModelType.vit_h) -> bytes:
        image_embedding = await self.get_image_embeddings(image_bytes, model_type)
        with tempfile.NamedTemporaryFile(prefix=f"embeddings_", suffix=".onnx", mode="r+b",
                                         delete=True) as embedding_file:
            np.save(embedding_file.file, image_embedding)
            embedding_file.flush()
            embedding_file.seek(0, 0)
            return embedding_file.read()

    async def generate_mask(self, image_bytes: bytes, model_type: SAMModelType, input_points: List[Point]):
        buffer = np.asarray(bytearray(image_bytes), dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        predictor = self._get_predictor(model_type)
        predictor.set_image(image)

        point_cords = []
        point_lables = []
        predictor.predict()

