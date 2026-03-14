import enum


class QueryKind(str, enum.Enum):
    ORGANIC = "organic"
    SCORING = "scoring"
