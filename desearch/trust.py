"""What validators take as given about the task API without asking it, pinned in each release so a change shows in review.

Until the API runs in an attested enclave (block D), the key that signs its log is pinned here instead of read from the API's own
`/v1/key`, where whoever answers the domain could name any key."""

# The key the task API signs its log, its notes beside uploads and its published files with.
TASK_API_SIGNER = "5CD6xo6ZmkFxWEwPbPaUh9cdznuk67EN48d6E5PuXFLyJ4so"
# The hotkey whose commitment on the subnet anchors the log's head: a key of its own, registered on the subnet, never the
# signer above (the API refuses to anchor with its signing key or the legacy signer, and validators refuse to read anchors
# from either). Empty until the anchor hotkey is registered and pinned here (docs/ops-checklist.md); until then anchors are
# not checked.
TASK_API_ANCHOR_HOTKEY = ""
