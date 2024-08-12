import io
import os.path
import tempfile
import warnings
from logging import Logger

import arrow
import cv2
import numpy as np
import torch
import torchvision
from onnxruntime.quantization import quantize_dynamic, QuantType

from api.config.settings import Settings
from api.model.sam import SAMModelType
from segment_anything import sam_model_registry, SamPredictor
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
        cuda_is_available = torch.cuda.is_available()
        print("CUDA is available:", cuda_is_available)

        device_type = "cpu"
        # device_type = "cuda" if cuda_is_available else "cpu"

        checkpoint = os.path.join(self.checkpoint_dir, self.model_types[default_model_type])

        print(f"Loading SAM model {checkpoint} in {device_type} mode ...")

        sam = sam_model_registry[default_model_type.value](checkpoint=checkpoint)
        sam.to(device=device_type)
        self.predictor = SamPredictor(sam)

    async def export_onnx_model(self, model_type: SAMModelType, quantize: bool = True):
        checkpoint = os.path.join(self.checkpoint_dir, self.model_types[model_type])
        sam = sam_model_registry[model_type.value](checkpoint=checkpoint)
        onnx_model = SamOnnxModel(sam, return_single_mask=True)

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

                with tempfile.NamedTemporaryFile(prefix=f"onnx_model_{model_type.value}", suffix=".onnx", mode="wb",
                                                 delete=False) as model_file:
                    model_file.write(exported.getvalue())
                    model_file.flush()

                try:
                    time_stamp = int(arrow.Arrow.now().float_timestamp * 1000)
                    onnx_model_quantized_path = f"sam_onnx_quantized_{model_type.value}_{time_stamp}.onnx"
                    quantize_dynamic(
                        model_input=model_file.name,
                        model_output=onnx_model_quantized_path,
                        per_channel=False,
                        reduce_range=False,
                        weight_type=QuantType.QUInt8,
                    )
                    with open(onnx_model_quantized_path, "rb") as output_file:
                        return output_file.read()
                finally:
                    if os.path.exists(model_file.name):
                        os.remove(model_file.name)

    async def get_image_embeddings(self, image_bytes: bytes) -> np.ndarray:
        buffer = np.asarray(bytearray(image_bytes), dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        self.predictor.set_image(image)
        image_embedding = self.predictor.get_image_embedding().cpu().numpy()
        # np.save("embedding_file.npy", image_embedding)
        return image_embedding

    async def get_image_embeddings_bytes(self, image_bytes: bytes) -> bytes:
        image_embedding = await self.get_image_embeddings(image_bytes)
        return image_embedding.tobytes()
