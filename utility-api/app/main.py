from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request

from app.config import CORS_ALLOWED_ORIGINS, NETUID, SUBTENSOR_NETWORK
from app.domains.logs.router import router as logs_router
from app.domains.miners.router import router as miners_router
from app.logger import get_logger
from app.redis_client import close_redis, init_redis

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info(
        f"Starting utility API lifespan: netuid={NETUID} "
        f"subtensor_network={SUBTENSOR_NETWORK}"
    )

    await init_redis()

    yield

    # Shutdown
    logger.info("Stopping utility API lifespan")
    await close_redis()


app = FastAPI(
    title="SN22 Utility API",
    description="Subnet-22 (Desearch) dataset & logging utility API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(logs_router)
app.include_router(miners_router)


@app.middleware("http")
async def log_unhandled_request_errors(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception:
        client = request.client.host if request.client else ""
        logger.exception(
            f"Unhandled request error: method={request.method} "
            f"path={request.url.path} "
            f"query={request.url.query} "
            f"hotkey={request.headers.get('X-Hotkey', '')} "
            f"client={client}"
        )
        raise


@app.get("/")
async def root():
    return {"message": "Subnet-22 utility api is running!"}
