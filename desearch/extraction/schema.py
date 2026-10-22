import hashlib

import pyarrow as pa

ERRORS = frozenset(
    {
        "timeout",
        "dns",
        "connect",
        "tls",
        "too_large",
        "not_html",
        "empty",
        "http_4xx",
        "http_5xx",
        "blocked",
        "redirect_loop",
        "other",
    }
)

PAGE_SCHEMA = pa.schema(
    [
        pa.field("url", pa.string(), nullable=False),
        pa.field("final_url", pa.string(), nullable=False),
        pa.field("status", pa.int32(), nullable=False),
        pa.field("error", pa.string(), nullable=True),
        pa.field("fetched_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("elapsed_ms", pa.int32(), nullable=False),
        pa.field("content_type", pa.string(), nullable=False),
        pa.field("html_bytes", pa.int32(), nullable=False),
        pa.field("html", pa.large_binary(), nullable=True),
        pa.field("html_sha256", pa.string(), nullable=False),
        pa.field("page_type", pa.string(), nullable=False),
        pa.field("title", pa.string(), nullable=False),
        pa.field("description", pa.string(), nullable=False),
        pa.field("lang", pa.string(), nullable=False),
        pa.field("canonical", pa.string(), nullable=False),
        pa.field("published", pa.string(), nullable=False),
        pa.field("author", pa.string(), nullable=False),
        pa.field("json_ld_types", pa.list_(pa.string()), nullable=False),
        pa.field("headings", pa.list_(pa.string()), nullable=False),
        pa.field("text", pa.large_string(), nullable=False),
        pa.field("text_sha256", pa.string(), nullable=False),
    ]
)


PAGE_SCHEMA_V2 = pa.schema(
    [
        *PAGE_SCHEMA,
        pa.field("extraction", pa.string(), nullable=False),
        pa.field("text_mode", pa.string(), nullable=False),
        pa.field("modified", pa.string(), nullable=False),
        pa.field("site_name", pa.string(), nullable=False),
        pa.field("image", pa.string(), nullable=False),
        pa.field("section", pa.string(), nullable=False),
        pa.field("tags", pa.list_(pa.string()), nullable=False),
        pa.field("breadcrumbs", pa.list_(pa.string()), nullable=False),
        pa.field("entity", pa.large_string(), nullable=False),
        pa.field("ld", pa.large_string(), nullable=False),
        pa.field("paywalled", pa.bool_(), nullable=True),
        pa.field("robots", pa.list_(pa.string()), nullable=False),
    ]
)


def sha256_hex(data: bytes | str | None) -> str:
    if data is None:
        return ""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()
