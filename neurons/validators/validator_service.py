import os
from typing import Optional

from pydantic import BaseModel

os.environ["USE_TORCH"] = "1"

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from desearch import QUERY_MINERS
from neurons.validators.validator import Neuron

neuron = Neuron()


@asynccontextmanager
async def lifespan(app):
    # Start the neuron when the app starts
    await neuron.start()
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
    if not neuron.available_uids:
        raise HTTPException(
            status_code=500,
            detail="Neuron is not available.",
        )

    config = neuron.config
    return config


class GetRandomUidRequest(BaseModel):
    # Specific UID to request
    uid: Optional[int] = None


@app.post(
    "/uid/random",
)
async def get_random_uid(body: GetRandomUidRequest):
    if not neuron.available_uids:
        raise HTTPException(
            status_code=500,
            detail="Neuron is not available.",
        )

    if body.uid is not None and body.uid in neuron.available_uids:
        return {"uid": body.uid, "axon": neuron.metagraph.axons[body.uid]}

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

    return {"uid": uid, "axon": axon}


@app.get("/")
async def health():
    if not neuron.available_uids:
        raise HTTPException(
            status_code=500,
            detail="No available UIDs.",
        )

    return {"status": "healthy"}


PORT = 8006

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, timeout_keep_alive=300)
