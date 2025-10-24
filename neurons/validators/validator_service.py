import os

os.environ["USE_TORCH"] = "1"

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from neurons.validators.validator import Neuron
from desearch import QUERY_MINERS


neuron = Neuron()


@asynccontextmanager
async def lifespan(app):
    # Start the neuron when the app starts
    await neuron.run()
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/config")
async def get_config():
    config = neuron.config
    return config


@app.get(
    "/uid/random",
)
async def get_random_uid():
    if not neuron.available_uids:
        raise HTTPException(
            status_code=500,
            detail="Neuron is not available.",
        )

    uid = await neuron.get_uids(
        strategy=QUERY_MINERS.RANDOM,
        is_only_allowed_miner=False,
        specified_uids=None,
    )

    if uid is None:
        raise HTTPException(
            status_code=500,
            detail="No available UID found.",
        )

    axon = neuron.metagraph.axons[uid]

    return {
        "uid": uid,
        "axon": axon,
    }


PORT = 8006

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, timeout_keep_alive=300)
