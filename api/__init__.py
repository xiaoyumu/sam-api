import logging
import os
from logging import Logger

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware

from api.config.settings import Settings
from api.controller.sam_controller import SAMController


def init_logger(settings: Settings) -> Logger:
    log_level = settings.log_level.upper()
    logging.basicConfig(format='[%(asctime)s.%(msecs)03d][%(levelname)s] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=log_level)
    logger = logging.getLogger(settings.app_name)
    return logger


def init_middleware(app: FastAPI, settings: Settings):
    app.add_middleware(CORSMiddleware,
                       allow_origins=settings.cors_allow_origins.split(","),
                       allow_credentials=settings.cors_allow_credentials,
                       allow_methods=settings.cors_allow_methods.split(","),
                       allow_headers=settings.cors_allow_headers.split(","))
    app.add_middleware(GZipMiddleware)


def init_controller(app: FastAPI, settings: Settings, logger: Logger):
    app.state.sam_controller = SAMController(settings, logger)


def init_view(app: FastAPI):
    from api.view.default import router as default_router
    app.include_router(default_router, prefix=app.state.settings.prefix)

    from api.view.sam import router as sam_router
    app.include_router(sam_router, prefix=app.state.settings.prefix)


def init_app():
    settings = Settings()
    logger = init_logger(settings)

    app = FastAPI(title=settings.app_name, version=settings.app_version, root_path=settings.openapi_prefix)
    app.state.settings = settings
    app.state.logger = logger

    init_middleware(app, settings)
    init_controller(app, settings, logger)
    init_view(app)

    return app
