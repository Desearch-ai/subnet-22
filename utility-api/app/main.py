from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.requests import Request

from app.config import NETUID, SUBTENSOR_NETWORK
from app.domains.dataset.router import close_question_cache, init_question_cache
from app.domains.dataset.router import router as dataset_router
from app.domains.logs.router import router as logs_router
from app.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info(
        f"Starting utility API lifespan: netuid={NETUID} "
        f"subtensor_network={SUBTENSOR_NETWORK}"
    )

    await init_question_cache(
        netuid=NETUID,
        subtensor_network=SUBTENSOR_NETWORK,
    )

    yield

    # Shutdown
    logger.info("Stopping utility API lifespan")
    await close_question_cache()


app = FastAPI(
    title="SN22 Utility API",
    description="Subnet-22 (Desearch) dataset & logging utility API",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(dataset_router)
app.include_router(logs_router)


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
