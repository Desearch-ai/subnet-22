from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import NETUID, SUBTENSOR_NETWORK
from app.domains.dataset.router import close_question_cache, init_question_cache
from app.domains.dataset.router import router as dataset_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_question_cache(
        netuid=NETUID,
        subtensor_network=SUBTENSOR_NETWORK,
    )

    yield

    # Shutdown
    await close_question_cache()


app = FastAPI(
    title="SN22 Utility API",
    description="Subnet-22 (Desearch) dataset & logging utility API",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(dataset_router)


@app.get("/")
async def root():
    return {"message": "Subnet-22 utility api is running!"}
