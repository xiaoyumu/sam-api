import uvicorn

from api.config.settings import Settings

if __name__ == '__main__':
    settings = Settings()
    uvicorn.run("api:init_app",
                host=settings.host,
                port=settings.port,
                timeout_keep_alive=settings.timeout_keep_alive,
                factory=True)
