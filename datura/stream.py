import asyncio
import random
import time
import bittensor as bt
from datura.protocol import ScraperStreamingSynapse


async def collect_response(response: ScraperStreamingSynapse, uid, start_time):
    async for chunk in response:
        if isinstance(chunk, bt.Synapse):
            end_time = time.time()
            duration = end_time - start_time
            process_time = chunk.dendrite.process_time
            if process_time is not None:
                print(
                    f"Miner uid {uid} finished with final synapse after {duration:.2f}s from start time. Dendrite process time: {process_time:.2f}s"
                )
            else:
                print(
                    f"Miner uid {uid} finished with final synapse after {duration:.2f}s from start time. Dendrite process time is None"
                )
            return chunk
    return None


async def collect_responses(async_responses, uids, start_time):
    tasks = [
        asyncio.create_task(collect_response(resp, uid, start_time))
        for resp, uid in zip(async_responses, uids)
    ]

    return await asyncio.gather(*tasks)


async def collect_final_synapses(async_responses, uids, start_time, group_size=15):
    final_synapses = [None] * len(async_responses)

    # Process all async_responses in sequence of groups
    # Split the async_responses into groups of size group_size
    async_responses_groups = [
        async_responses[i : i + group_size]
        for i in range(0, len(async_responses), group_size)
    ]

    group_indices = list(range(len(async_responses_groups)))
    random.shuffle(group_indices)

    for group_index in group_indices:
        async_responses_group = async_responses_groups[group_index]
        group_uids = uids[group_index * group_size : (group_index + 1) * group_size]

        group_final_synapses = await collect_responses(
            async_responses_group, group_uids, start_time
        )

        for i, synapse in enumerate(group_final_synapses):
            final_synapses[group_index * group_size + i] = synapse

    return final_synapses
