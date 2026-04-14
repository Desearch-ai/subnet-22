import argparse
import asyncio

import bittensor as bt
import uvicorn
from fastapi import FastAPI
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
async def ai_search(synapse: ScraperStreamingSynapse):
    return JSONStreamResponse(synapse)


@app.post("/twitter/search")
async def twitter_search(synapse: TwitterSearchSynapse):
    synapse = await twitter_search_miner.search(synapse)
    return synapse.model_dump()


@app.post("/twitter/id")
async def tweet_by_id(synapse: TwitterIDSearchSynapse):
    synapse = await twitter_search_miner.search_by_id(synapse)
    return synapse.model_dump()


@app.post("/twitter/urls")
async def tweet_by_urls(synapse: TwitterURLsSearchSynapse):
    synapse = await twitter_search_miner.search_by_urls(synapse)
    return synapse.model_dump()


@app.post("/web/search")
async def web_search(synapse: WebSearchSynapse):
    synapse = await web_search_miner.search(synapse)
    return synapse.model_dump()


def parse_args():
    parser = argparse.ArgumentParser(description="Desearch Miner Worker API")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--timeout-keep-alive", type=int, default=300)
    return parser.parse_args()


async def main():
    args = parse_args()
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
