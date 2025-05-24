import time
import torch
import random
from datura.redis.redis_client import redis_client
import jsonpickle


class OrganicHistoryMixin:
    HISTORY_EXPIRY_TIME = 2 * 3600

    def __init__(self):
        self.organic_history = {}
        self._history_loaded = False

    @property
    def redis_key(self):
        return f"{self.__class__.__name__}:organic_history"

    async def _ensure_history_loaded(self):
        """Ensure history is loaded from Redis"""
        if not self._history_loaded:
            self.organic_history = await self._load_history()
            self._history_loaded = True

    async def _load_history(self):
        data = await redis_client.get(self.redis_key)
        if data:
            decoded_data = jsonpickle.decode(data)
            return {int(uid): values for uid, values in decoded_data.items()}
        return {}

    async def _save_history(self, history):
        await redis_client.set(
            self.redis_key, jsonpickle.encode(history), ex=self.HISTORY_EXPIRY_TIME
        )

    async def _clean_organic_history(self):
        await self._ensure_history_loaded()

        current_time = time.time()
        self.organic_history = {
            uid: [
                value
                for value in values
                if value["start_time"] >= current_time - self.HISTORY_EXPIRY_TIME
            ]
            for uid, values in self.organic_history.items()
        }

        self.organic_history = {
            uid: values
            for uid, values in self.organic_history.items()
            if len(values) > 0
        }

        await self._save_history(self.organic_history)

        return self.organic_history

    async def _save_organic_response(
        self, uids, responses, tasks, event, start_time
    ) -> None:
        await self._ensure_history_loaded()

        for uid, response, task, *event_values in zip(
            uids, responses, tasks, *event.values()
        ):
            event = dict(zip(event.keys(), event_values))

            if uid.item() not in self.organic_history:
                self.organic_history[uid.item()] = []

            self.organic_history[uid.item()].append(
                {
                    "response": response,
                    "task": task,
                    "event": event,
                    "start_time": start_time,
                }
            )

        await self._save_history(self.organic_history)

    async def get_random_organic_responses(self):
        await self._clean_organic_history()

        event = {}
        tasks = []
        responses = []
        uids = []

        for uid, item in self.organic_history.items():
            uids.append(torch.tensor([uid]))

            random_index = random.randint(0, len(item) - 1)

            responses.append(item[random_index]["response"])
            tasks.append(item[random_index]["task"])
            for key, value in item[random_index]["event"].items():
                if not key in event:
                    event[key] = []

                event[key].append(value)

        return {
            "event": event,
            "tasks": tasks,
            "responses": responses,
            "uids": torch.tensor(uids),
        }

    async def get_latest_organic_responses(self):
        await self._clean_organic_history()

        event = {}
        tasks = []
        responses = []
        uids = []

        for uid, item in self.organic_history.items():
            uids.append(torch.tensor([uid]))
            responses.append(item[-1]["response"])
            tasks.append(item[-1]["task"])
            for key, value in item[-1]["event"].items():
                if not key in event:
                    event[key] = []

                event[key].append(value)

        return {
            "event": event,
            "tasks": tasks,
            "responses": responses,
            "uids": torch.tensor(uids),
        }

    async def get_uids_with_no_history(self, available_uids):
        await self._clean_organic_history()

        uids = [uid for uid in available_uids if uid not in self.organic_history]

        return uids
