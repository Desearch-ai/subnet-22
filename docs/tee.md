# The task API in a TEE

The task API runs as a [dstack](https://github.com/Dstack-TEE/dstack) app in an Intel TDX confidential
VM (CVM). This page is the contract between the API, validators and anyone checking it: what the API
proves about itself, in which formats, and how a new version takes over. It is defense in depth, not
the root of trust: validators replay the API's signed log (block B of the trust design), so a broken
TEE can still be caught. What the TEE adds is that the team cannot quietly run other code, change a
setting, edit Redis, or sign with a key of its own.

- [What is pinned where](#what-is-pinned-where)
- [Formats](#formats)
- [Approving a version](#approving-a-version)
- [Custody: which key validators follow](#custody-which-key-validators-follow)
- [Pause](#pause)
- [Chain inputs](#chain-inputs)
- [What is left](#what-is-left)

## What is pinned where

| What | Pinned by | Checked by |
| --- | --- | --- |
| The source | a git tag `task-api-v*` | anyone |
| The image | its digest, built from the tag by [CI](../.github/workflows/task-api-image.yml) with a signed SLSA provenance linking commit and digest; the build is reproducible | `gh attestation verify`, or rebuilding it ([verify](./verify-task-api.md)) |
| The compose | [`task-api/deploy/dstack/docker-compose.yml`](../task-api/deploy/dstack/docker-compose.yml) names every image by `@sha256:` and writes every setting; `app-compose.json` around it lists only credentials in `allowed_envs`. Its SHA-256 is the **compose hash** | the TDX quote's RTMR3 (the `compose-hash` event) |
| The OS | dstack's production image (no SSH, no debug), its MRTD and RTMR0–2 | validators pin them in their release ([`desearch/attestation.py`](../desearch/attestation.py)) |
| The hardware | genuine TDX, an up-to-date TCB, the device ids allowed | Intel's DCAP signature chain ([dcap-qvl](https://github.com/Phala-Network/dcap-qvl)); `sha256(PPID)` against the release's device allowlist |
| The signing key | generated inside the CVM at first boot, kept on its encrypted disk, never leaves it except to an approved successor | bound into the quote's `report_data` |
| The version in charge | validators' stake, on chain | the API itself before a handover, and every validator |

Redis runs inside the CVM with no port outside it. SQLite lives on the CVM's encrypted volume. TLS ends
inside the CVM: the dstack gateway passes the TLS stream through (`<app_id>-443s.<gateway>`), and the
certificate's public key is bound into the quote, so a client can tell it is talking to the attested
CVM and not to a proxy.

## Formats

All JSON that is hashed or signed is in canonical form: `json.dumps(value, sort_keys=True,
separators=(",", ":"), ensure_ascii=False)` encoded as UTF-8, which is what the API's
[`py::canonical`](../task-api/src/py.rs) writes. Hashes and keys are lowercase hex without `0x`.
Signatures are sr25519 over the canonical bytes, as every other signature of the API.

### `GET /v1/config`

Every setting the running API uses, by environment variable name, with secrets left out (their names
are listed under `secrets_set` when set, never their values):

```json
{"version": "0.1.0", "commit": "<git sha the image was built from, or null>",
 "settings": {"TASK_API_CLAIM_TTL": 180.0, "TASK_API_KEY_SOURCE": "cvm", "...": "..."},
 "secrets_set": ["CF_R2_ACCESS_KEY_ID", "CF_R2_SECRET_ACCESS_KEY"]}
```

A test in the API (`the_measured_compose_writes_every_setting_the_config_shows`) fails when the dstack
compose leaves any of these to a default, so the compose alone decides the answer.

`config_sha256 = sha256(canonical(config))` is bound into the attestation, so the answer can be checked
against the quote. The answer also has `paused`, the pause or `null` (see [Pause](#pause)); it is state, not
a setting, so it is left out of what the attestation binds.

**What a CVM refuses to start with.** With `TASK_API_KEY_SOURCE=cvm` the API checks its settings together
before serving and stops with the setting to fix: the chain read through at least 2 endpoints that must agree
(`TASK_API_CHAIN_QUORUM`), no one operator running that many of them (an endpoint's operator is the last two
labels of its host name, or its address and port), finalized heads only and never counted on the clock, the
log anchored (`TASK_API_ANCHOR_NETWORK`) through its own quorum of at least 2 under the same rule, by
`TASK_API_ANCHOR_KEY_URI`, a hotkey that is neither the log's signing key nor the legacy signer, each pinned
upload locked for at least 86,400 s (`TASK_API_UPLOAD_LOCK_S`, the day it can still be checked or taken back
in), and `TASK_API_GOV_MIN_COLDKEYS` of at least 1. Validators check the same in the compose
([`MINIMUM_SETTINGS`](../desearch/attestation.py)), so a compose that turns a protection off is never
approved.

### `GET /v1/attestation`

```json
{
  "document": {
    "kind": "sn22-task-api-attestation",
    "version": 1,
    "signer": "<ss58 of the signing key>",
    "handover_key": "<x25519 public key the CVM receives a predecessor's state with>",
    "config_sha256": "<sha256 of canonical /v1/config>",
    "tls_spki_sha256": "<sha256 of the DER SubjectPublicKeyInfo of the served TLS certificate, or null>",
    "nonce": "<hex, only when asked with ?nonce=>"
  },
  "report_data": "<128 hex chars>",
  "quote": "<hex TDX quote v4>",
  "event_log": [{"imr": 3, "event_type": 134217729, "digest": "<hex>", "event": "compose-hash", "event_payload": "<hex>"}],
  "vm_config": "<string, as dstack returns it>",
  "app_compose": "<the exact app-compose.json bytes the CVM booted, as a string>",
  "info": {"app_id": "...", "instance_id": "...", "compose_hash": "...", "os_image_hash": "...", "device_id": "..."}
}
```

- `report_data = sha512(b"sn22-task-api-attestation-v1:" + canonical(document))`, all 64 bytes.
- The quote's `report_data` field (the last 64 bytes of the TD report body) must equal it.
- `info` is what dstack says about itself; it is a convenience and proves nothing. Everything that
  matters is taken from the quote and the event log.
- `?nonce=<up to 64 hex chars>` returns a fresh quote over a document with that nonce, to prove the key
  is held by a live CVM now. At most one is made at a time.

To check an attestation (what [`tools/verify_attestation.py`](../tools/verify_attestation.py) and every
validator do):

1. Verify the quote's signature chain to Intel's root with DCAP collateral; require TCB status
   `UpToDate` (the release may allow listed advisories) and the TD's debug attribute off.
2. Require MRTD, RTMR0, RTMR1 and RTMR2 to be one of the pinned production OS measurements.
3. Replay the event log's IMR 3 events: start from 48 zero bytes, `rtmr = sha384(rtmr || digest)`
   for each event, where a dstack runtime event (`event_type` `0x08000001`) has
   `digest = sha384(event_type as 4 bytes little-endian || b":" || event || b":" || payload)` (V1),
   or `sha384(JCS({"name", "payload": hex, "type"}))` when the event says `"version": 2`. The result
   must equal RTMR3.
4. Take `compose-hash`, `app-id`, `instance-id` and `key-provider` from the replayed events. Require
   `sha256(app_compose) == compose-hash`, and the app compose to pass the policy:
   `runner` `docker-compose`, `kms_enabled` true, `local_key_provider_enabled` false, no
   `pre_launch_script` and no `init_script`, `allowed_envs` within the release's credential list,
   every image in `docker_compose_file` pinned by `@sha256:`, no `privileged`, no host network or
   ports other than the API's.
5. Require `sha256(PPID)` (dstack's device id) to be in the release's device allowlist, when it has one.
6. Require the quote's `report_data` to equal the formula above, and `document.signer` to be the signer
   being checked.

### Approval commitments

A validator approves a version with its Bittensor commitment on netuid 22 (one per hotkey, the newest
replaces the last). The committed bytes are UTF-8, split into `Raw` fields of at most 128 bytes, read
back concatenated:

```
sn22tee:<compose hash>                      approves one version
sn22tee:<compose hash>,<compose hash>       approves two, the running one and its successor
sn22tee:<compose hash>@<signer ss58>        approves a version and appoints one of its CVMs (genesis, recovery)
```

[`tools/approve_version.py`](../tools/approve_version.py) writes one. Anything else in a commitment is
ignored.

### Custody records

The chain of keys the log has been signed with. Each record is a JSON object with a `signature`:

```json
{"kind": "appoint", "seq": 0, "prev": null,
 "signer": "<ss58>", "compose_hash": "<hex>", "attestation_sha256": "<sha256 of canonical document, no nonce>",
 "upload_log_seq": 81200, "block": 4012345, "at": 1762300000.0, "signature": "<hex>"}

{"kind": "handover", "seq": 1, "prev": "<record hash of seq 0>",
 "from": "<ss58 of the signer handing over>", "signer": "<ss58 of the successor>",
 "compose_hash": "<successor's>", "attestation_sha256": "<successor's>", "state_sha256": "<hex>",
 "upload_log_seq": 81234, "block": 4060000, "at": 1762900000.0, "signature": "<hex>"}
```

- `upload_log_seq` is the last upload log file the previous holder signed: files up to it are signed by
  the previous holder, files after it by `signer`. A manifest frozen at or before `block` may carry the
  previous holder's signature; later ones only `signer`'s. Before the first record the holder is the
  release's legacy signer (block B pins it).
- Record 0 is either an `appoint` or a `handover` from that legacy signer (`TASK_API_LEGACY_SIGNER` in
  the compose): the genesis, which moves today's state into the first CVM.

- An `appoint` record is signed by its own `signer`; a `handover` by `from`, which must be the signer of
  the record before it.
- `record hash = sha256(canonical(record with its signature))`; `prev` links each record to the one
  before, and `seq` counts from 0.
- `block` is the finalized block the approval was checked at.
- The API writes them to the uploads bucket as `log/custody/seq/<seq, 12 digits>.json`, the newest one
  again as `log/custody/latest.json`, and each signer's attestation as
  `log/custody/attestations/<ss58>.json`. `GET /v1/custody` serves the records this copy holds with its
  status (`standby`, `active`, `handing_over`, `retired`) and the pause.

## Approving a version

All of it is read at one block. The **independent validators** are the hotkeys with a validator permit
whose coldkey is not the subnet owner's (`SubtensorModule.SubnetOwner`), that are not the owner's hotkey
(`SubnetOwnerHotkey`), and none of whose parents (`SubtensorModule.ParentKeys` on the subnet) is the
owner's hotkey or a hotkey the owner's coldkey owns (`SubtensorModule.Owner`): a child of the owner
inherits the owner's stake. A validator's **weight** is its `total_stake` in the metagraph less what the
owner's coldkey holds on it, counted the way the metagraph counts it: its alpha on the subnet plus its root
TAO times the TAO weight (`StakeInfoRuntimeApi.get_stake_info_for_coldkey(owner)` and
`SubtensorModule.TaoWeight` as a fraction of `u64::MAX`, rounded down), so stake the owner delegates to an
independent validator counts for nobody.

- **support(H, N, age)** — the weight of independent validators whose commitment at block N lists H and
  was made at least `age` blocks before N; **coldkeys(H, N, age)** — how many distinct coldkeys they have.
- **H is approved at N** — `support(H, N, timelock) > threshold × total independent weight`, compared in
  whole numbers: `support × 1,000,000 > round(threshold × 1,000,000) × total`; at 0.5, `support × 2 > total`;
  and `coldkeys(H, N, timelock) ≥ min_coldkeys`. With no independent stake nothing is approved.
- **S is appointed for H at N** — the same, counting only commitments `sn22tee:H@S`.
- **H is still endorsed at N** — the same with no timelock (`age` 0), so moving one's commitment from `H`
  to `H,H'` keeps the running version endorsed while `H'` waits.

The defaults are `threshold = 0.5`, `min_coldkeys = 3` and `timelock = 50400` blocks (7 days). The API
reads them from the compose (`TASK_API_GOV_THRESHOLD`, `TASK_API_GOV_MIN_COLDKEYS`,
`TASK_API_GOV_TIMELOCK_BLOCKS`) and reads the commitments, the metagraph, the owner keys, the owner's stake
and the validators' parents itself, at a finalized block its chain quorum agrees on; validators pin the
same values in their release and refuse an API whose compose says otherwise (they also pin
`TASK_API_KEY_SOURCE=cvm`, `TASK_API_CUSTODY=1`, `TASK_API_CHAIN_FINALIZED=1`,
`TASK_API_CHAIN_COUNT_ON=0` and `TASK_API_TEE_SIMULATED=0`). The owner cannot shorten the timelock: it is
part of the measured compose and of the validators' release.

**What this does not stop: the Bittensor baseline.** The owner can buy stake under coldkeys nobody can tie
to it and run validators with them; those are independent here, as they are to Yuma consensus. The rule
cannot tell them apart from anyone else's stake, only make the owner put up real stake for each, under at
least three coldkeys, for the whole timelock, in public. The trust this assumes is the one Bittensor
already rests on: that most of a subnet's stake is not one party's.

## Custody: which key validators follow

At any block validators follow exactly one signer: the `signer` of the last valid custody record, while
its version is still endorsed.

- **Genesis and recovery.** A CVM of an approved version starts in standby, with a fresh key nobody
  else has. Validators who checked its attestation commit `sn22tee:H@S`. Once S is appointed, the CVM
  writes an `appoint` record (with `prev` the last record, if any) and starts serving. A recovery
  record replaces whatever signer came before, so a CVM that is lost for good can be replaced, but only
  through the same stake and timelock. A recovering CVM continues the log rather than starting it again:
  - it reads the whole custody chain from the bucket, checking every link and signature, so it knows
    every key that held the log;
  - it takes the published log head (`log/journal/latest.json`), which must be signed by the last holder
    and reach at least the entry the chain anchors for the anchor hotkey (read through the anchor quorum):
    a head behind the anchor means files are missing, and the CVM stays in standby;
  - it replays the published files with the API's own rules from the newest snapshot before both the head
    and the anchored entry, every file checked for a holder's signature and its links from the one before,
    and the anchored entry's hash against the chain; any break leaves it in standby;
  - it takes the accounts, the rule-time clock and the pause the replay arrives at, adopts the head (its
    first entry, `custody`, names that head as `prev`, and the record's `journal_seq` is its number),
    and logs a snapshot of the accounts. A validator's replay goes on through the `custody` entry and
    compares that snapshot with its own accounts: a recovering CVM cannot hand back budgets, strikes,
    lockouts or credits the log does not give.
  The claims, uploads and rounds the lost holder had in flight are lost with it, and the replay drops
  them too; the claims cost their miners nothing. To keep recovery possible the API anchors only heads it
  has published.
- **Handover.** The admin asks the running CVM to hand over to a successor's URL. The running CVM
  - fetches the successor's attestation and checks it as above, from inside the TEE, with its own OS
    measurements (or those its compose allows for a successor) as the pinned ones;
  - checks on chain that the successor's compose hash is approved;
  - stops taking work, snapshots SQLite and Redis, encrypts the snapshot to the successor's attested
    `handover_key` (X25519, HKDF-SHA256, ChaCha20-Poly1305), and sends it;
  - after the successor confirms with a signature, writes and publishes the `handover` record and
    retires for good: from then on it signs nothing.
  The successor restores the snapshot and starts serving once it sees the published record naming it.
  If the handover fails before the record is published, the running CVM takes up work again and the
  successor stays in standby, so there are never two signers.
  On the wire: `POST /v1/admin/handover {"successor": "<url>"}`, signed by an admin hotkey, on the
  running CVM; it sends `POST /v1/custody/receive` to the successor with four bytes of big-endian length,
  the JSON `{"record", "chain", "seal"}`, then the sealed state (`seal` is the ephemeral X25519 key and
  the 12-byte nonce; the AEAD's associated data is the record hash; at most 4 GiB). The successor answers
  `{"ack": <signature over canonical {"ack": record hash}>}`. Every write but these two is refused with
  `503` while a copy does not hold the log, and a claim gets the `UNAVAILABLE` refusal.
- **Genesis from today's server.** The non-TEE server hands over the same way, as the legacy signer:
  the first CVM takes a `handover` at seq 0 only from `TASK_API_LEGACY_SIGNER`. That one step trusts the
  old server's state, as everyone does today; from then on every holder is attested.
- **Validators** check every new record: the signature, the link, the approval at its block (from
  their own node, an archive node for old blocks), and the new signer's attestation against the
  record's `compose_hash` and `attestation_sha256`. They keep what they accepted. A different record
  at a `seq` they already accepted is a fork: they stop following, keep both signed records as proof,
  and wait for the record anchored on chain (block B) to settle it.
- **One signer, even with two genuine copies.** A key exists in one CVM only (it is random, not
  derived from the app's KMS key, which every copy of an app shares). A second copy of the same version
  has a different key, and no record names it unless validators appoint it or the current signer hands
  over to it. The current signer hands over once and then refuses to sign; a copy restored from an old
  disk finds a later record naming someone else, in the bucket or anchored on chain, and stays retired.

## Pause

The only emergency power. `POST /v1/admin/pause` with `{"paused": true, "reason": "..."}`, signed by an
admin hotkey, stops the accounts until `{"paused": false}`:

- claims are answered with the refusal `PAUSED` (signed into the log like every refusal, once a minute
  per miner); completions, abandons and validators' reports with `503` and `Retry-After`, a refused
  completion of a held claim logged with the miner's signed request; no upload is settled and no verdict
  finalized, so nobody is paid or penalized while it lasts;
- the pause is an outage of cause `pause`: every claim alive in it lapses at no cost (`pause_outage`),
  decided again by a replay from the logged outage;
- the pause, the outage's start (or end) and the `pause` entry with the admin's signed request are
  written in one transaction; the pause is kept in SQLite, so every copy and a restart see it, and a CVM
  appointed without a handover takes it from the replayed log;
- it is shown at `/v1/config` (`paused`) and `/v1/custody`, and a replay flags (suspect) a completion,
  abandon, report, settlement or verdict logged while the log shows a pause.

A pause changes no code, setting, key or account.

## Chain inputs

The API reads the chain from several independent RPC endpoints (`TASK_API_NETWORK`, comma-separated)
and takes a block height or hash only when `TASK_API_CHAIN_QUORUM` of them agree. Anchoring reads and
writes through its own list and quorum (`TASK_API_ANCHOR_NETWORK`, `TASK_API_ANCHOR_QUORUM`): the genesis
hash, runtime versions and nonce its signature covers and the anchor it reads back are taken only when that
many agree, and every anchor is sent to every endpoint, so one endpoint that lies, listed first or not,
cannot make anchors fail their signature check or keep them off the chain. With
`TASK_API_CHAIN_FINALIZED=1` it reads the finalized head, not the best block, so a seed never comes from
a block that can still change. With `TASK_API_CHAIN_COUNT_ON=0` it never counts blocks on its own clock:
the current block is the last one the quorum agreed on, so a host that delays the network can only make
time pass slower, which gives miners more time, never less. The endpoints are reached over TLS from
inside the CVM, so the host cannot answer for them.

## What is left

- A GRANDPA light client inside the CVM, to replace the RPC quorum: it would need no trusted endpoint
  at all. Validators check every block hash the log used against their own node, so it is not needed for
  them. The compose's quorum is two of two operators (the Opentensor Foundation's endpoint and
  OnFinality's public one): either one failing stalls the chain reads, which forgives the claims alive then
  and pays nobody until it is back; a third operator's endpoint would let the quorum ride out one.
- `TASK_API_TEE_SIMULATED=1` lets the dstack simulator stand in on a testnet: a successor's quote is then
  taken without Intel's signature. Validators refuse any compose with it on.
- Recovery rebuilds the accounts from the published log, so it is only as complete as the bucket: a
  holder lost with entries it never published loses them (it never anchors them, so the chain cannot
  disagree).
- dstack does not stop a host from restoring an old disk of the same CVM. The custody records and their
  on-chain anchors (block B) make such a copy harmless, not impossible.
- Physical attacks on TDX (TEE.fail, 2025) can forge quotes. That is why the TEE is not the root of
  trust: validators replay the log.
