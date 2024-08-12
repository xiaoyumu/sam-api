from starlette.requests import Request

from api.controller.sam_controller import SAMController


def sam_controller(req: Request) -> SAMController:
    return req.app.state.sam_controller
