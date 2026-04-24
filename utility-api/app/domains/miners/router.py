from app.config import VALIDATOR_URLS
from app.domains.miners.client import fetch_all_miners, fetch_miner_detail
from app.domains.miners.schemas import (
    MinerDetail,
    MinerDetailResponse,
    MinerListItem,
    MinerListResponse,
    MinerTypeState,
    ScoringWindow,
    ValidatorMinerView,
)
from app.logger import get_logger
from fastapi import APIRouter, HTTPException, Path

router = APIRouter(prefix="/miners", tags=["miners"])
logger = get_logger(__name__)


@router.get("", response_model=MinerListResponse)
async def list_miners():
    """Aggregate `/public/miners` from every configured validator.

    Validators are queried in parallel. The response includes one
    `ValidatorInfo` per configured validator (online or not) plus a list
    of miners, each carrying a per-validator map of current per-search-type
    state.
    """

    if not VALIDATOR_URLS:
        raise HTTPException(status_code=503, detail="No validators configured")

    results = await fetch_all_miners(VALIDATOR_URLS)

    validators = [info for info, _ in results]
    by_miner: dict[str, dict] = {}

    for info, data in results:
        if not data:
            continue
        for m in data.get("miners", []):
            entry = by_miner.setdefault(
                m["hotkey"],
                {
                    "hotkey": m["hotkey"],
                    "uid": m["uid"],
                    "coldkey": m["coldkey"],
                    "by_validator": {},
                },
            )
            entry["by_validator"][info.id] = {
                st: MinerTypeState(**state) for st, state in m["per_type"].items()
            }

    miners = [MinerListItem(**v) for v in by_miner.values()]
    miners.sort(key=lambda m: m.uid)

    logger.info(
        f"/miners served: configured={len(VALIDATOR_URLS)} "
        f"online={sum(1 for v in validators if v.online)} "
        f"miners={len(miners)}"
    )

    return MinerListResponse(validators=validators, miners=miners)


@router.get("/{hotkey}", response_model=MinerDetailResponse)
async def get_miner_detail(
    hotkey: str = Path(..., description="Miner hotkey (ss58)"),
):
    """Fan out `/public/miners/{hotkey}` to every configured validator in parallel.

    Returns one view per validator. `detail` is null if the validator was
    unreachable or doesn't know the miner.
    """

    if not VALIDATOR_URLS:
        raise HTTPException(status_code=503, detail="No validators configured")

    results = await fetch_miner_detail(VALIDATOR_URLS, hotkey)

    views: list[ValidatorMinerView] = []
    for info, data in results:
        miner = (data or {}).get("miner") if data else None
        if not miner:
            views.append(ValidatorMinerView(validator=info, detail=None))
            continue

        per_type = {
            st: MinerTypeState(**state) for st, state in miner["per_type"].items()
        }
        windows = {
            st: [ScoringWindow(**w) for w in ws]
            for st, ws in (miner.get("windows") or {}).items()
        }
        detail = MinerDetail(
            hotkey=miner["hotkey"],
            uid=miner["uid"],
            coldkey=miner["coldkey"],
            per_type=per_type,
            windows=windows,
        )
        views.append(ValidatorMinerView(validator=info, detail=detail))

    if all(v.detail is None for v in views):
        raise HTTPException(status_code=404, detail="Miner not found")

    return MinerDetailResponse(views=views)
