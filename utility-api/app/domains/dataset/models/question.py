import uuid

from sqlalchemy import Column, DateTime, Enum, String, Text, func
from sqlalchemy.dialects.postgresql import ARRAY, UUID

from app.db.base import Base
from app.domains.dataset.enums import AISearchTool, SearchType


class Question(Base):
    """
    Stores dataset questions used by SN22 validators.

    - search_types:    which search modes this question applies to
                       (ai, x, x-post lookups, web).
    - ai_search_tools: relevant AI-search tools (arxiv, wikipedia, …). Nullable
                       — only meaningful when ai_search is in search_types.
    - source:          origin of the question, e.g. "huggingface:squad",
                       "desearch", "manual".
    """

    __tablename__ = "questions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    query = Column(Text, nullable=False)

    search_types = Column(
        ARRAY(Enum(SearchType, name="search_type_enum", create_constraint=False)),
        nullable=False,
    )

    ai_search_tools = Column(
        ARRAY(Enum(AISearchTool, name="ai_search_tool_enum", create_constraint=False)),
        nullable=True,
    )

    source = Column(String, nullable=True)

    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<Question {self.id} query={self.query[:40]!r}>"
