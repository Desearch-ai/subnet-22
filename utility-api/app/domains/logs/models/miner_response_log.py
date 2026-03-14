import uuid

from app.db.base import Base
from app.domains.dataset.enums import SearchType
from app.domains.logs.enums import QueryKind
from sqlalchemy import Column, DateTime, Enum, Float, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID


class MinerResponseLog(Base):
    __tablename__ = "miner_response_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    query_kind = Column(
        Enum(QueryKind, name="query_kind_enum", create_constraint=False),
        nullable=False,
    )
    search_type = Column(
        Enum(SearchType, name="search_type_enum", create_constraint=False),
        nullable=False,
    )
    netuid = Column(Integer, nullable=False)

    scoring_epoch_start = Column(DateTime(timezone=True), nullable=True)

    miner_uid = Column(Integer, nullable=True)
    miner_hotkey = Column(String, nullable=False)
    miner_coldkey = Column(String, nullable=True)

    validator_uid = Column(Integer, nullable=True)
    validator_hotkey = Column(String, nullable=False)
    validator_coldkey = Column(String, nullable=True)

    request_query = Column(Text, nullable=False)
    status_code = Column(Integer, nullable=True)
    process_time = Column(Float, nullable=True)
    total_reward = Column(Float, nullable=True)

    response_payload = Column(JSONB, nullable=False)
    reward_payload = Column(JSONB, nullable=True)

    __table_args__ = (
        Index(
            "ix_miner_response_logs_miner_hotkey_created_at",
            "miner_hotkey",
            "created_at",
        ),
        Index(
            "ix_miner_response_logs_miner_uid_created_at",
            "miner_uid",
            "created_at",
        ),
        Index(
            "ix_miner_response_logs_query_kind_search_type_created_at",
            "query_kind",
            "search_type",
            "created_at",
        ),
    )
