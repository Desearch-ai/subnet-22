import bittensor as bt
from starlette.types import Send
from desearch.protocol import (
    DeepResearchSynapse,
    ReportItem,
)
import json


class DeepResearchMiner:
    def __init__(self, miner: any):
        self.miner = miner

    async def deep_research(self, synapse: DeepResearchSynapse, send: Send):
        prompt = synapse.prompt
        tools = synapse.tools
        date_filter = synapse.date_filter_type
        system_message = synapse.system_message

        bt.logging.debug(
            f"DeepResearchMiner: prompt: {prompt}, tools: {tools}, date_filter: {date_filter}, system_message: {system_message}"
        )

        sections = [
            {
                "title": "Introduction",
                "description": "Blockchain technology has emerged as a revolutionary framework for secure and transparent data transactions. Its decentralized nature fundamentally alters how digital information is managed and shared across various sectors, extending beyond its initial applications in cryptocurrency. This report delves into the essential components of blockchain, the role of consensus mechanisms, and the broader ecosystem, providing a comprehensive understanding of blockchain technology.",
                "links": [],
            },
            {
                "title": "Essential Components of Blockchain",
                "description": "Description for Essential Components of Blockchain",
                "links": [],
                "subsections": [
                    {
                        "title": "1. Nodes",
                        "description": "Nodes are integral to the functioning of a blockchain network. Each node is a participant that maintains a copy of the blockchain and is responsible for validating transactions. Nodes relay transaction data across the network, ensuring that all copies of the decentralized ledger are synchronized and up to date. The decentralized nature of nodes contributes to the overall security and resilience of the blockchain, as it prevents single points of failure.",
                        "links": [
                            "https://www.geeksforgeeks.org/components-of-blockchain-network/"
                        ],
                    },
                    {
                        "title": "2. Decentralized Ledgers",
                        "description": "At the heart of blockchain technology lies the decentralized ledger, which records all activities and transactions in a tamper-proof manner. Unlike traditional ledgers controlled by a single entity, a decentralized ledger is shared across all nodes in the network, enhancing transparency and trust among participants. Each transaction is cryptographically secured, further ensuring the integrity of the data recorded.",
                        "links": [
                            "https://www.geeksforgeeks.org/components-of-blockchain-network/"
                        ],
                    },
                    {
                        "title": "3. Consensus Mechanisms",
                        "description": "Consensus mechanisms are crucial for achieving agreement among network participants regarding the validity of transactions before they are added to the ledger. These mechanisms prevent fraud and maintain the integrity of the blockchain. Various consensus models exist, each with its unique algorithms and processes, which can significantly impact the blockchain's security and efficiency.",
                        "links": [
                            "https://accountend.com/understanding-the-3-key-components-of-blockchain-technology/"
                        ],
                    },
                ],
            },
        ]

        # list of sources used for report
        await send(
            {
                "type": "http.response.body",
                "body": json.dumps(
                    {
                        "type": "search",
                        "content": [
                            {
                                "title": "Blockchain",
                                "link": "https://www.geeksforgeeks.org/components-of-blockchain-network/",
                                "snippet": "Buy Bitcoin, Ethereum, and other leading cryptocurrencies on a platform trusted by millions.",
                            }
                        ],
                    }
                ).encode("utf-8"),
                "more_body": True,
            }
        )

        # return streaming chunks of report content in md format
        await send(
            {
                "type": "http.response.body",
                "body": json.dumps(
                    {
                        "type": "report_md",
                        "content": "chunks for report content in md format",
                    }
                ).encode("utf-8"),
                "more_body": True,
            }
        )

        report = [ReportItem(**section).model_dump() for section in sections]

        await send(
            {
                "type": "http.response.body",
                "body": json.dumps({"type": "report", "content": report}).encode(
                    "utf-8"
                ),
                "more_body": False,
            }
        )
