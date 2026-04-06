import asyncio
import hashlib
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Sequence

SCORING_DATASET_NAME = "sentence-transformers/natural-questions"
SCORING_DATASET_CONFIG = "pair"
SCORING_DATASET_SPLIT = "train"
SCORING_DATASET_COLUMN = "query"
SCORING_DATASET_FALLBACK_COLUMNS: tuple[str, ...] = ("question",)
SCORING_DATASET_MAX_ROWS: int | None = None
SCORING_CONCURRENCY = 10

SCORING_SEARCH_TYPES: tuple[str, ...] = ("ai_search", "x_search", "web_search")

AI_SEARCH_TOOLS: list[list[str]] = [
    ["Twitter Search", "Reddit Search"],
    ["Twitter Search", "Web Search"],
    ["Twitter Search", "Web Search"],
    ["Twitter Search", "Web Search"],
    ["Twitter Search", "Web Search"],
    ["Twitter Search", "Hacker News Search"],
    ["Twitter Search", "Hacker News Search"],
    ["Twitter Search", "Youtube Search"],
    ["Twitter Search", "Youtube Search"],
    ["Twitter Search", "Youtube Search"],
    ["Twitter Search", "Web Search"],
    ["Twitter Search", "Reddit Search"],
    ["Twitter Search", "Reddit Search"],
    ["Twitter Search", "Hacker News Search"],
    ["Twitter Search", "ArXiv Search"],
    ["Twitter Search", "ArXiv Search"],
    ["Twitter Search", "Wikipedia Search"],
    ["Twitter Search", "Wikipedia Search"],
    ["Twitter Search", "Web Search"],
    ["Twitter Search", "Web Search"],
    ["Twitter Search", "Web Search"],
    ["Web Search"],
    ["Reddit Search"],
    ["Hacker News Search"],
    ["Youtube Search"],
    ["ArXiv Search"],
    ["Wikipedia Search"],
    ["Twitter Search", "Youtube Search", "ArXiv Search", "Wikipedia Search"],
    ["Twitter Search", "Web Search", "Reddit Search", "Hacker News Search"],
    [
        "Twitter Search",
        "Web Search",
        "Reddit Search",
        "Hacker News Search",
        "Youtube Search",
        "ArXiv Search",
        "Wikipedia Search",
    ],
]

AI_SEARCH_DATE_FILTERS: list[str] = (
    ["PAST_24_HOURS"] * 4
    + ["PAST_2_DAYS"] * 5
    + ["PAST_WEEK"] * 5
    + ["PAST_2_WEEKS"] * 5
    + ["PAST_MONTH"]
    + ["PAST_YEAR"]
)

X_SEARCH_PARAM_FIELDS: tuple[str, ...] = (
    "sort",
    "is_quote",
    "is_video",
    "is_image",
    "min_retweets",
    "min_replies",
    "min_likes",
    "date_range",
)

THREE_YEARS_IN_DAYS = 3 * 365


def current_scoring_window() -> datetime:
    now = datetime.now(timezone.utc)
    return now.replace(minute=0, second=0, microsecond=0)


def derive_deterministic_int(*parts: Any, bits: int = 63) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    value = int.from_bytes(digest[:8], "big")
    mask = (1 << bits) - 1
    return value & mask


def normalize_questions(rows: Iterable[Any]) -> list[str]:
    unique_questions: list[str] = []
    seen: set[str] = set()

    for row in rows:
        if row is None:
            continue

        question = str(row).strip()
        if not question or question in seen:
            continue

        seen.add(question)
        unique_questions.append(question)

    return unique_questions


def generate_ai_search_params(rng: random.Random) -> dict[str, Any]:
    return {
        "tools": list(rng.choice(AI_SEARCH_TOOLS)),
        "date_filter_type": rng.choice(AI_SEARCH_DATE_FILTERS),
    }


def generate_x_search_params(
    rng: random.Random,
    *,
    base_time: datetime,
) -> dict[str, Any]:
    selected_field = rng.choice(X_SEARCH_PARAM_FIELDS)
    params: dict[str, Any] = {}

    if selected_field == "sort":
        params["sort"] = "Latest"
    elif selected_field == "date_range":
        end_date = base_time - timedelta(days=rng.randint(0, THREE_YEARS_IN_DAYS))
        start_date = end_date - timedelta(days=rng.randint(7, 14))
        params["start_date"] = start_date.strftime("%Y-%m-%d_%H:%M:%S_UTC")
        params["end_date"] = end_date.strftime("%Y-%m-%d_%H:%M:%S_UTC")
    elif selected_field == "is_video":
        params["is_video"] = rng.choice([True, False])
    elif selected_field == "is_image":
        params["is_image"] = rng.choice([True, False])
    elif selected_field == "is_quote":
        params["is_quote"] = rng.choice([True, False])
    elif selected_field == "min_likes":
        params["min_likes"] = rng.randint(5, 100)
    elif selected_field == "min_replies":
        params["min_replies"] = rng.randint(5, 20)
    elif selected_field == "min_retweets":
        params["min_retweets"] = rng.randint(5, 20)

    return params


def generate_params_for(
    search_type: str,
    *,
    window_seed: int,
    time_range_start: datetime,
    uid: int | None = None,
) -> dict[str, Any]:
    if search_type == "ai_search":
        rng = random.Random(
            derive_deterministic_int(
                window_seed, time_range_start.isoformat(), search_type
            )
        )
        return generate_ai_search_params(rng)

    if search_type == "x_search":
        rng = random.Random(
            derive_deterministic_int(
                window_seed,
                time_range_start.isoformat(),
                search_type,
                uid,
            )
        )
        return generate_x_search_params(
            rng,
            base_time=time_range_start.astimezone(timezone.utc),
        )

    return {}


@dataclass(frozen=True)
class ScoringQuestion:
    query: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoringAssignment:
    time_range_start: datetime
    uid: int
    search_type: str
    validator_uid: int
    validator_hotkey: str
    question: ScoringQuestion
    scoring_seed: int


class HuggingFaceQuestionPool:
    def __init__(
        self,
        *,
        dataset_name: str,
        dataset_config: str | None,
        split: str,
        question_column: str,
        fallback_question_columns: Sequence[str] = ("question",),
        max_rows: int | None = None,
    ):
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.split = split
        self.question_column = question_column
        self.fallback_question_columns = tuple(fallback_question_columns)
        self.max_rows = max_rows
        self._questions: list[str] = []

    @property
    def questions(self) -> list[str]:
        return list(self._questions)

    async def initialize(self) -> None:
        if self._questions:
            return

        self._questions = await asyncio.to_thread(self._load_questions)

    def _load_questions(self) -> list[str]:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "The 'datasets' package is required for validator-local scoring "
                "datasets. Install it before running the validator."
            ) from exc

        load_args: list[str] = [self.dataset_name]
        if self.dataset_config:
            load_args.append(self.dataset_config)

        dataset = load_dataset(*load_args, split=self.split)

        candidate_columns = (self.question_column, *self.fallback_question_columns)
        selected_column = next(
            (column for column in candidate_columns if column in dataset.column_names),
            None,
        )
        if selected_column is None:
            raise ValueError(
                "Unable to find a question column in the dataset. "
                f"Tried {candidate_columns}, available columns are {dataset.column_names}."
            )

        rows = dataset[selected_column]
        if self.max_rows is not None:
            rows = rows[: self.max_rows]

        questions = normalize_questions(rows)
        if not questions:
            raise ValueError(
                f"Dataset {self.dataset_name!r} did not yield any usable questions."
            )

        return questions

    def get_questions_for(self, search_type: str) -> list[str]:
        if search_type not in SCORING_SEARCH_TYPES:
            raise ValueError(f"Unsupported search type: {search_type}")
        if not self._questions:
            raise RuntimeError("Question pool is not initialized yet.")
        return self.questions


def build_question_pool() -> HuggingFaceQuestionPool:
    return HuggingFaceQuestionPool(
        dataset_name=SCORING_DATASET_NAME,
        dataset_config=SCORING_DATASET_CONFIG,
        split=SCORING_DATASET_SPLIT,
        question_column=SCORING_DATASET_COLUMN,
        fallback_question_columns=SCORING_DATASET_FALLBACK_COLUMNS,
        max_rows=SCORING_DATASET_MAX_ROWS,
    )


def build_scoring_assignments(
    *,
    time_range_start: datetime,
    miner_uids: Sequence[int],
    validators: Sequence[Any],
    question_pool: HuggingFaceQuestionPool,
    combined_seed: int,
) -> list[ScoringAssignment]:
    if not miner_uids or not validators:
        return []

    assignments: list[ScoringAssignment] = []
    ordered_miner_uids = sorted(int(uid) for uid in miner_uids)

    validator_ownership = build_validator_ownership(
        time_range_start=time_range_start,
        miner_uids=ordered_miner_uids,
        validators=validators,
        combined_seed=combined_seed,
    )

    for search_type in SCORING_SEARCH_TYPES:
        questions = question_pool.get_questions_for(search_type)
        question_order = list(range(len(questions)))
        random.Random(
            derive_deterministic_int(
                combined_seed,
                time_range_start.isoformat(),
                search_type,
                "question-order",
            )
        ).shuffle(question_order)

        for index, uid in enumerate(ordered_miner_uids):
            owner_validator = validator_ownership[uid]
            question_index = question_order[index % len(question_order)]
            query = questions[question_index]
            params = generate_params_for(
                search_type,
                window_seed=combined_seed,
                time_range_start=time_range_start,
                uid=uid,
            )
            scoring_seed = derive_deterministic_int(
                combined_seed,
                time_range_start.isoformat(),
                search_type,
                uid,
                "scoring-seed",
            )
            assignments.append(
                ScoringAssignment(
                    time_range_start=time_range_start,
                    uid=uid,
                    search_type=search_type,
                    validator_uid=int(owner_validator.uid),
                    validator_hotkey=owner_validator.hotkey,
                    question=ScoringQuestion(query=query, params=params),
                    scoring_seed=scoring_seed,
                )
            )

    task_order_rng = random.Random(
        derive_deterministic_int(
            combined_seed, time_range_start.isoformat(), "task-order"
        )
    )
    task_order_rng.shuffle(assignments)
    return assignments


def build_validator_ownership(
    *,
    time_range_start: datetime,
    miner_uids: Sequence[int],
    validators: Sequence[Any],
    combined_seed: int,
) -> dict[int, Any]:
    if not miner_uids or not validators:
        return {}

    ordered_validators = sorted(
        validators, key=lambda item: (int(item.uid), item.hotkey)
    )

    validator_order = list(ordered_validators)

    random.Random(
        derive_deterministic_int(
            combined_seed,
            time_range_start.isoformat(),
            "validator-order",
        )
    ).shuffle(validator_order)

    miner_order = sorted(int(uid) for uid in miner_uids)
    random.Random(
        derive_deterministic_int(
            combined_seed,
            time_range_start.isoformat(),
            "miner-order",
        )
    ).shuffle(miner_order)

    ownership: dict[int, Any] = {}
    for index, uid in enumerate(miner_order):
        ownership[uid] = validator_order[index % len(validator_order)]

    return ownership


def filter_scoring_assignments(
    assignments: Sequence[ScoringAssignment],
    *,
    validator_uid: int,
) -> list[ScoringAssignment]:
    return [item for item in assignments if item.validator_uid == int(validator_uid)]
