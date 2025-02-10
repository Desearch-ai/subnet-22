import time


class OrganicHistoryMixin:
    def __init__(self):
        self.organic_history = {}
        self.HISTORY_EXPIRY_TIME = 4 * 3600

    def _clean_organic_history(self):
        current_time = time.time()
        self.organic_history = {
            uid: value
            for uid, value in self.organic_history.items()
            if value["start_time"] >= current_time - self.HISTORY_EXPIRY_TIME
        }

    def _merge_synthetic_organic_responses(
        self, responses, uids, tasks, event, start_time, available_uids
    ):
        responses_dict = {}
        for response, uid, task, *event_values in zip(
            responses, uids, tasks, *event.values()
        ):
            event = dict(zip(event.keys(), event_values))

            responses_dict[uid.item()] = {
                "response": response,
                "task": task,
                "event": event,
            }

        merged_uids = []
        merged_responses = []
        merged_tasks = []
        merged_event = {key: [] for key in event}

        for uid in available_uids:
            item = (
                self.organic_history.pop(uid.item())
                if uid.item() in self.organic_history
                else responses_dict.get(uid.item(), None)
            )

            if item:
                merged_uids.append(uid)
                merged_responses.append(item.response)
                merged_tasks.append(item.task)
                for key, value in item.event.items():
                    merged_event[key].append(value)

        return merged_event, merged_tasks, merged_responses, merged_uids, start_time

    def _save_organic_response(self, uids, responses, tasks, event, start_time) -> None:
        for uid, response, task, event_values in zip(
            uids, responses, tasks, *event.values()
        ):
            event = dict(zip(event.keys(), event_values))

            self.organic_history[uid.item()] = {
                "response": response,
                "task": task,
                "event": event,
                "start_time": start_time,
            }
