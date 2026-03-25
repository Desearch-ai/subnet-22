import traceback
import time
import bittensor as bt
from starlette.types import Send
from desearch.protocol import (
    ScraperStreamingSynapse,
)
from desearch.tools.tool_manager import ToolManager
from desearch.dataset.date_filters import (
    DateFilter,
    DateFilterType,
    get_specified_date_filter,
)
from desearch.utils import get_max_execution_time
from datetime import datetime
import pytz


class ScraperMiner:
    def __init__(self, miner: any):
        self.miner = miner

    async def smart_scraper(self, synapse: ScraperStreamingSynapse, send: Send):
        try:
            prompt = synapse.prompt

            bt.logging.trace(synapse)

            bt.logging.info(
                "================================== Prompt ==================================="
            )
            bt.logging.info(prompt)
            bt.logging.info(
                "================================== Prompt ===================================="
            )

            date_filter = get_specified_date_filter(DateFilterType.PAST_2_WEEKS)

            if synapse.start_date and synapse.end_date and synapse.date_filter_type:
                start_date = datetime.strptime(
                    synapse.start_date, "%Y-%m-%dT%H:%M:%SZ"
                ).replace(tzinfo=pytz.utc)

                end_date = datetime.strptime(
                    synapse.end_date, "%Y-%m-%dT%H:%M:%SZ"
                ).replace(tzinfo=pytz.utc)

                date_filter = DateFilter(
                    start_date=start_date,
                    end_date=end_date,
                    date_filter_type=DateFilterType(synapse.date_filter_type),
                )

            if synapse.max_execution_time is None:
                synapse.max_execution_time = get_max_execution_time(
                    synapse.model, synapse.count or 10
                )

            tool_manager = ToolManager(
                synapse=synapse,
                date_filter=date_filter,
                send=send,
                start_time=time.time(),
            )

            await tool_manager.run()

            bt.logging.info("End of Streaming")

        except Exception as e:
            bt.logging.error(f"error in scraper miner {e}\n{traceback.format_exc()}")
