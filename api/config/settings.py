from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "SegmentAnything Backend Service"
    app_version: str = "0.0.1"
    timeout_keep_alive: int = 30
    host: str = "localhost"
    port: int = 8080
    prefix: str = ""
    openapi_prefix: str = ""
    log_level: str = "info"

    cors_allow_origins: str = "*"
    cors_allow_credentials: bool = True
    cors_allow_methods: str = "*"
    cors_allow_headers: str = "*"


