import argparse
import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

import bittensor as bt
import uvicorn
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response

from desearch.protocol import (
    ScraperStreamingSynapse,
    TwitterIDSearchSynapse,
    TwitterSearchSynapse,
    TwitterURLsSearchSynapse,
    WebSearchSynapse,
)
from neurons.miners.scraper_miner import ScraperMiner
from neurons.miners.twitter_search_miner import TwitterSearchMiner
from neurons.miners.web_search_miner import WebSearchMiner
from neurons.miners.worker_auth import verify_worker_request

bt.logging.on()
bt.logging.set_info(True)

app = FastAPI(title="Desearch Miner Worker API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

scraper_miner = ScraperMiner(miner=None)
twitter_search_miner = TwitterSearchMiner(miner=None)
web_search_miner = WebSearchMiner(miner=None)


class JSONStreamResponse(Response):
    media_type = "application/json"

    def __init__(self, synapse: ScraperStreamingSynapse):
        super().__init__(content=b"", media_type=self.media_type)
        self.synapse = synapse

    async def __call__(self, scope, receive, send):
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    [b"content-type", self.media_type.encode("utf-8")],
                    [b"cache-control", b"no-cache"],
                    [b"connection", b"keep-alive"],
                ],
            }
        )

        await scraper_miner.smart_scraper(self.synapse, send)


@app.get("/")
async def read_root():
    return {"message": "Miner worker API is running"}


@app.post("/ai/search")
async def ai_search(
    synapse: ScraperStreamingSynapse,
    _: str = Depends(verify_worker_request),
):
    return JSONStreamResponse(synapse)


@app.post("/twitter/search")
async def twitter_search(
    synapse: TwitterSearchSynapse,
    _: str = Depends(verify_worker_request),
):
    synapse = await twitter_search_miner.search(synapse)
    return synapse.model_dump()


@app.post("/twitter/id")
async def tweet_by_id(
    synapse: TwitterIDSearchSynapse,
    _: str = Depends(verify_worker_request),
):
    synapse = await twitter_search_miner.search_by_id(synapse)
    return synapse.model_dump()


@app.post("/twitter/urls")
async def tweet_by_urls(
    synapse: TwitterURLsSearchSynapse,
    _: str = Depends(verify_worker_request),
):
    synapse = await twitter_search_miner.search_by_urls(synapse)
    return synapse.model_dump()


@app.post("/web/search")
async def web_search(
    synapse: WebSearchSynapse,
    _: str = Depends(verify_worker_request),
):
    synapse = await web_search_miner.search(synapse)
    return synapse.model_dump()


def parse_args():
    parser = argparse.ArgumentParser(description="Desearch Miner Worker API")
    parser.add_argument(
        "--host", type=str, default=os.environ.get("WORKER_HOST", "0.0.0.0")
    )
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("WORKER_PORT", "8000"))
    )
    parser.add_argument("--timeout-keep-alive", type=int, default=300)
    parser.add_argument(
        "--wallet.name",
        dest="wallet_name",
        type=str,
        default=os.environ.get("WALLET_NAME", "miner"),
    )
    parser.add_argument(
        "--wallet.hotkey",
        dest="wallet_hotkey",
        type=str,
        default=os.environ.get("WALLET_HOTKEY", "default"),
    )
    parser.add_argument(
        "--subtensor.network",
        dest="network",
        type=str,
        default=os.environ.get("SUBTENSOR_NETWORK", "test"),
    )
    parser.add_argument(
        "--netuid", type=int, default=int(os.environ.get("NETUID", "41"))
    )
    return parser.parse_args()


async def main():
    from neurons.miners import worker_auth

    args = parse_args()

    wallet = bt.Wallet(name=args.wallet_name, hotkey=args.wallet_hotkey)
    miner_hotkey = wallet.hotkey.ss58_address

    worker_auth.init(miner_hotkey, args.network)
    await worker_auth.init_metagraph(args.network, args.netuid)
    asyncio.create_task(worker_auth.sync_metagraph_loop())

    config = uvicorn.Config(
        app,
        host=args.host,
        port=args.port,
        timeout_keep_alive=args.timeout_keep_alive,
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
