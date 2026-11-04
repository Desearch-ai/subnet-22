"""What validators take as given about the task API without asking it, pinned in each release so a change shows in review.

Until the API runs in an attested enclave (block D), the key that signs its log is pinned here instead of read from the API's own
`/v1/key`, where whoever answers the domain could name any key."""

# The key the task API signs its log, its notes beside uploads and its published files with.
TASK_API_SIGNER = "5CD6xo6ZmkFxWEwPbPaUh9cdznuk67EN48d6E5PuXFLyJ4so"
# The hotkey whose commitment on the subnet anchors the log's head; the API commits with the key it signs with.
TASK_API_ANCHOR_HOTKEY = TASK_API_SIGNER
