"""Reading an upload no validator checked: only the columns published, in byte ranges, refused before decoding when too large."""

from __future__ import annotations

import io
from collections.abc import Callable

import pyarrow.parquet as pq

MAGIC = b"PAR1"
MAX_FOOTER_BYTES = 2_000_000
MAX_ROW_BYTES = 6_000_000
MAX_ROW_GROUP_BYTES = 512_000_000


class RangeFile(io.RawIOBase):
    """A remote file read in byte ranges, so a reader fetches only the parts it decodes."""

    def __init__(self, size: int, read_range: Callable[[int, int], bytes]):
        self.size = size
        self.read_range = read_range
        self.position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.position = offset
        elif whence == io.SEEK_CUR:
            self.position += offset
        else:
            self.position = self.size + offset
        return self.position

    def readinto(self, buffer) -> int:
        end = min(self.size, self.position + len(buffer))
        if end <= self.position:
            return 0
        data = self.read_range(self.position, end - 1)
        buffer[: len(data)] = data
        self.position += len(data)
        return len(data)


def footer_fits(source: RangeFile) -> bool:
    """Both magic markers present and a footer small enough to parse, read from the last 8 bytes."""
    if source.size < 12:
        return False
    head = source.read_range(0, 3)
    tail = source.read_range(source.size - 8, source.size - 1)
    if head != MAGIC or tail[4:] != MAGIC:
        return False
    length = int.from_bytes(tail[:4], "little")
    return 0 < length <= min(MAX_FOOTER_BYTES, source.size - 12)


def too_big(metadata, columns: list[str], assigned: int) -> bool:
    """Checked on the footer, before decompressing anything."""
    wanted = set(columns)
    groups = [
        sum(
            group.column(c).total_uncompressed_size
            for c in range(metadata.num_columns)
            if group.column(c).path_in_schema.split(".")[0] in wanted
        )
        for group in (metadata.row_group(g) for g in range(metadata.num_row_groups))
    ]
    rows = max(assigned, 1)
    return (
        metadata.num_rows > 2 * rows
        or sum(groups) > rows * MAX_ROW_BYTES
        or max(groups, default=0) > MAX_ROW_GROUP_BYTES
    )


def read_rows(
    source: RangeFile, columns: list[str], assigned: int
) -> list[dict] | None:
    """The given columns of every row, or None when the file cannot be decoded safely."""
    try:
        if not footer_fits(source):
            return None
        parquet = pq.ParquetFile(source, pre_buffer=True)
        if too_big(parquet.metadata, columns, assigned):
            return None
        table = parquet.read(columns=columns)
        # The footer is the miner's word; the decoded columns are not.
        if table.nbytes > max(assigned, 1) * MAX_ROW_BYTES:
            return None
        return table.to_pylist()
    except Exception:
        return None
