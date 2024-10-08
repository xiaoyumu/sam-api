from http import HTTPStatus
from typing import Annotated, List, Tuple

from arrow import arrow
from fastapi import APIRouter, Depends, UploadFile
from fastapi.params import Form
from starlette.responses import Response

from api.controller.sam_controller import SAMController
from api.depends.controller import sam_controller
from api.model.sam import OnnxModelExport, SAMModelType, Point

router = APIRouter(prefix="/sam", tags=["SAM"])


@router.post("/embeddings")
async def get_image_embeddings(
        image: UploadFile,
        mode_type: Annotated[SAMModelType, Form()] = SAMModelType.vit_h,
        ctrl: SAMController = Depends(sam_controller)):
    image_bytes = await image.read()
    start = arrow.Arrow.now()
    embeddings_bytes = await ctrl.get_image_embeddings_bytes(image_bytes, mode_type)
    end = arrow.Arrow.now()
    duration = str((end-start).total_seconds())
    file_name = f"embeddings_{arrow.Arrow.now().int_timestamp}.npy"
    return Response(status_code=HTTPStatus.OK,
                    content=embeddings_bytes,
                    headers={"Content-Type": "application/octet-stream",
                             "Content-Disposition": f'attachment;filename={file_name}',
                             "x-generation-time-s": duration})


@router.post("/onnx-model/export")
async def export_onnx_model(
        request: OnnxModelExport,
        ctrl: SAMController = Depends(sam_controller)
):
    start = arrow.Arrow.now()
    exported = await ctrl.export_onnx_model(request.source_model_type, request.quantized)
    end = arrow.Arrow.now()
    duration = str((end - start).total_seconds())
    suffix = "" if not request.quantized else ".quant"
    file_name = f"exported_{request.source_model_type.value}_{arrow.Arrow.now().int_timestamp}{suffix}.onnx"
    return Response(status_code=HTTPStatus.OK,
                    content=exported,
                    headers={"Content-Type": "application/octet-stream",
                             "Content-Disposition": f'attachment;filename={file_name}',
                             "x-generation-time-s": duration})


@router.post("/generate-mask")
async def generate_mask(
        image: UploadFile,
        mode_type: Annotated[SAMModelType, Form()] = SAMModelType.vit_h,
        input_points: Annotated[List[Point], Form()] = None,
        ctrl: SAMController = Depends(sam_controller)):
    image_bytes = await image.read()
    start = arrow.Arrow.now()
    mask = await ctrl.generate_mask(image_bytes, mode_type, input_points)
    end = arrow.Arrow.now()
    duration = str((end - start).total_seconds())

