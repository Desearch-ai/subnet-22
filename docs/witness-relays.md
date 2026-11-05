# Witness relays, private upload keys and locked uploads

Two ways the operator of the task API could hurt one miner without leaving a trace are closed here:

- **The network path (hole 8).** The API sits behind a network the operator runs. It could delay or drop one
  miner's completion until its claim expired (half its budget and a strike), or steer one miner's claims.
- **The storage account (hole 9).** The uploads bucket is the operator's. Writing over a miner's upload after
  its completion made validators read `412`, and they failed the miner for `upload_changed`.

What remains is a full outage, which is public and costs nobody anything (see [the canary](#storage-canary) and
[validators' storage witnesses](#storage-outages-validators-open)).

## Relays

A validator can serve a **relay**: the same three paths as the task API (`POST /v1/tasks/claim`,
`/v1/tasks/{id}/complete`, `/v1/tasks/{id}/abandon`). A miner sends its signed request there instead of to the
API; the relay

1. checks the miner's signature, its timestamp (60 s, as the API does) and that the hotkey is on the subnet, so
   it never forwards, or vouches for, a request the miner did not sign;
2. forwards the request **unchanged**: the four auth headers and the body byte for byte, no header of its own;
3. signs a **witness record** of it (below) and hands it back to the miner in the `X-Witness` header (base64
   JSON), whether or not the API answered. The record keeps what the API answered: the receipts it signed
   (a claim's, a completion's, an abandon's) or the refusal it gave, so the API is held to an outcome, not
   only to having been asked. Of a claim's answer only the receipts are kept, never the upload links.

The relay sees a request; it can't change it (the miner's signature covers the method, path, body, time and
nonce). It also sees the upload link in a claim's answer. On a versioned bucket that link is worth nothing to
anyone else, since the completion pins the miner's own version of its upload
([below](#uploads-pinned-by-version)). On a bucket without versions a relay could write over the key before the
completion; the API then refuses to pin an object whose entity tag is not the one the miner's upload got (409),
and the miner uploads once more. Until the uploads bucket is versioned, send claims through relays you trust.

**Witness record** (`desearch/witness.py`, kind `request`), signed by the relay's hotkey in `witness_signature`:

| Field | |
| --- | --- |
| `witness` | the relaying validator's hotkey |
| `request` | the request's hash: `sha256(signing payload + "\n" + hotkey + "\n" + signature)`, the same in the API (`task_api::auth::request_hash`) |
| `miner`, `method`, `path`, `body_sha256`, `timestamp`, `nonce`, `signature` | the miner's signed request, enough for anyone to check the miner's signature |
| `block` | the head of the validator's own node when it forwarded the request |
| `forwarded_at` | its clock then |
| `status`, `response_sha256`, `answered_ms`, `error` | what came back; `status` null when nothing did |
| `promised`, `receipts` | the signed receipts the API's answer carried, as it sent them; each names the request and the block it was judged at |
| `answer` | a refusal's status and the start of its detail |

Records signed before this field list (`version` 1) still verify against the fields they were signed over.

Each relay publishes its records at `GET /v1/witness/requests` once `--relay.publish_delay_blocks` blocks have
passed (5 by default), so nobody learns in real time which relay a miner uses. `GET /v1/witness/requests/{hash}`
answers at once, but only someone who saw the request can name its hash.

### Censorship proofs

Every receipt the API signs names the request it answers (`request` in the receipt body, the round log and
the hash-chained log) and the block it was judged at. A completion's answer carries its receipt and an
abandon's its receipts, as a claim's always did; and every completion or abandon the API turns down is logged
with why and the block it was judged at, accepted or not (past 30 refusals of one hotkey in a minute the rest
are answered 429 and not logged, so nobody grows the log at will).

Once `--relay.check_after_blocks` blocks (50) have passed and the anchored log covers them, every witnessed
request is held to its outcome. It is a **censorship proof** (`reason` in the proof) when:

| `reason` | |
| --- | --- |
| `unlogged` | the API signed a receipt the anchored log does not hold, or answered a completion, abandon or storage witness (accepted or refused) and logged nothing for it |
| `contradicted` | the log's entry for it says otherwise: another receipt, refused where the API answered with a receipt, accepted where it answered a refusal, or a different refusal |
| `late` | the log judged it more than `--relay.late_blocks` (10) blocks after the relay forwarded it: the API held it back, a claim running out meanwhile |
| `missing` | nothing came back (no answer within the relay's 90 s, or a gateway's 502, 504, 520-530) and the log never listed it among the requests it let in |

The miner signed the request, an independent validator forwarded it at a block of its own node, and the API's
own signature or its anchored log shows the rest. Receipts count only when a key this validator pins signed them
(the custody chain's holders, or `--relay.signer`). A repeated completion is answered with the first one's
receipt, which names the first request, and is held to that. Answers that leave no entry by design are owed
none: a claim answered without a receipt (a poll over its rate, a repeated refusal), and 401, 429, 500 and
503 answers; a selective 500 or 503 is visible in the witnesses but proves nothing on its own.

- Each validator checks its own records every minute (relayed requests, and its own storage witnesses) and
  keeps every proof, signed by it, at `GET /v1/witness/proofs` when it serves a relay; it logs each as an
  error. The fraud proofs its check of the API's log writes are published beside them, at
  `GET /v1/journal/proofs`. Outcome proofs need the anchored log, which every validator follows; the miner's tool below checks
  only that a request arrived.
- A miner can check the records its relays signed for it, kept in `WITNESSES_FILE`:

  ```bash
  python task-api/tools/check_witnesses.py --signer <the API's log key> --witnesses witnesses.jsonl --network finney
  ```

  It prints each proof as a JSON line and exits 1 if there is one. `--proofs FILE` checks proofs someone else
  made; `--relay URL` reads a relay's published records instead of a file.

A request sent through two relays is witnessed twice, by two independent validators, so a single relay that
drops requests and then claims censorship is outvoted by the other's record of the API's answer.

Until block B anchors the log on chain, "missing" means "no receipt naming it that verifies against the pinned
log key": a request listed by the API without such a receipt proves nothing either way, and an API that does not
answer lookups yields no proofs (it is down for everyone).

### Running a relay (validators)

| Flag | Default | |
| --- | --- | --- |
| `--relay.port` | 0 | serve the relay on this port; 0 serves none |
| `--relay.host` | `0.0.0.0` | |
| `--relay.publish_delay_blocks` | 5 | blocks before a record is listed publicly |
| `--relay.check_after_blocks` | 50 | blocks after which a request whose outcome the anchored log lacks or contradicts is a proof |
| `--relay.late_blocks` | 10 | blocks after forwarding within which the log must have judged a request |
| `--relay.signer` | the API's `/v1/key` | the log key proofs are checked against; pin it (block B pins it in the release) |

The relay serves at the root of its address (the miner signs the path, so it can't sit behind a prefix). Put it
behind your own TLS terminator if you like; a miner's limit is 2 requests a second with bursts of 10. Records and
proofs are kept a week in `witness.db` next to the ledger.

### Several paths (miners)

| Variable | Default | |
| --- | --- | --- |
| `RELAY_URLS` | none | comma-separated relay URLs; with any, each request goes over one path picked at random |
| `DIRECT_PATH` | 1 | 0 sends everything through relays, never straight to the API |
| `WITNESSES_FILE` | none | file each witness a relay signs for you is appended to |

A path that does not answer, answers 429 or a server error is passed over and rests (30 s, doubling to 10 min);
the next path gets the request, signed afresh. An answer from the API itself (409, 422, ...) is not a path
failing and is not retried elsewhere. Relays are listed by hand for now; announcing them on chain is a follow-up.

## Upload keys

A claim's upload key is `uploads/<32 random hex>.parquet` (`.f16` for embed tasks): 128 bits from the API's
CSPRNG, naming no miner, task or claim. It reaches the public log (the upload log, the open list, the signed
note) only with the completion. Storage, and the path to it, can't tell whose upload a write carries, so it
can't single one out; the canary below makes failing everyone at once public and harmless.

## Uploads pinned by version

On a bucket with versioning, every write to a key is a new version, and the completion pins one:

- the miner's upload answer names its version (`x-amz-version-id`) and entity tag; the miner sends both with
  the completion (`version`, `etag`), and the API pins exactly that version, so nothing anyone else wrote to the
  key, before or after, can take its place. Without a version, a completion whose `etag` differs from what the
  key holds is refused with 409 and the miner can upload again within its claim;
- the signed note beside the upload and the upload log name the `version`; on a bucket without versions the
  field is absent and the note signs the same bytes as before;
- validators read `uploads/...?versionId=<version>` and send no `If-Match`: a later write can't make the read
  fail, so an overwrite can no longer fail a miner. A version that is gone is `upload_missing` (void, no
  penalty), and a bucket that refuses reads by version is the validator's trouble, retried later, never the
  miner's;
- with `TASK_API_UPLOAD_LOCK_S` set, the API puts the pinned version under Object Lock in **compliance mode** for
  that long, so nobody, the bucket's owner included, can delete it or shorten the lock before the check window
  ends. A lock storage refuses fails the completion closed (503, complete again): nothing unlocked is pinned.
  The refusal opens an `upload_lock` outage, closed by the next lock that holds, so a claim that lapses while
  locks fail costs its miner nothing.

### Bucket configuration

Not applied to any live bucket; this is what the operator would set up, on a new bucket since most stores turn
Object Lock on only at creation (OVHcloud Object Storage and AWS S3 both do):

```bash
# a new uploads bucket with versioning and Object Lock, and no default retention: the API locks only the
# pinned upload versions, not the open list it rewrites every few seconds
aws s3api create-bucket --bucket subnet-22-v2 --object-lock-enabled-for-bucket --endpoint-url "$CF_R2_ENDPOINT"

# objects and old versions go a day after they stop being current; a locked version stays until its lock ends
aws s3api put-bucket-lifecycle-configuration --bucket subnet-22-v2 --endpoint-url "$CF_R2_ENDPOINT" \
  --lifecycle-configuration '{"Rules":[{"ID":"day","Status":"Enabled","Filter":{},
    "Expiration":{"Days":1},"NoncurrentVersionExpiration":{"NoncurrentDays":1},
    "AbortIncompleteMultipartUpload":{"DaysAfterInitiation":1}}]}'

# anyone may read objects, read uploads by version and see their lock
aws s3api put-bucket-policy --bucket subnet-22-v2 --endpoint-url "$CF_R2_ENDPOINT" --policy '{
  "Version":"2012-10-17","Statement":[
   {"Effect":"Allow","Principal":"*","Action":["s3:GetObject"],"Resource":"arn:aws:s3:::subnet-22-v2/*"},
   {"Effect":"Allow","Principal":"*","Action":["s3:GetObjectVersion","s3:GetObjectRetention"],
    "Resource":"arn:aws:s3:::subnet-22-v2/uploads/*"}]}'
```

Then `CF_R2_BUCKET=subnet-22-v2` and `TASK_API_UPLOAD_LOCK_S=86400` for the API, and the bucket's public URL as
validators' `--neuron.storage_url`. On a store that opens objects one by one (`CF_R2_PUBLIC_OBJECTS=1`, as on
OVHcloud), the API opens the pinned version itself. **Order:** validators must run this release before a
versioned bucket is switched on, since a note naming a `version` signs bytes older validators don't expect.
Cloudflare R2 has no object versions; its bucket locks refuse overwrites outright, which would also refuse a
miner's own retried upload, so R2 stays on entity tags.

## Storage observations

Every validator looks at every completed upload the upload log lists, through the public bucket as anyone
would (a `HEAD`, by version when pinned), and signs what it saw (`desearch/witness.py`, kind `storage`):
status, entity tag, version, size, lock mode and lock end, and whether that `matches` what the completion
pinned. A mismatch is logged as a warning. Records are kept a week and published at `GET /v1/witness/storage`
on a validator serving a relay; `--witness.storage_off` stops them. They are inputs for block B's replay:
what storage held is on record from many hands, not the bucket owner's word.

## Storage outages validators open

The canary below is the API's own word. Each validator also probes storage the way miners use it, from where it
stands: every `--witness.probe_every_s` (60 s) it asks the API for a probe link (`POST /v1/storage/probe-link`,
validators only: a presigned PUT, GET and DELETE of a fresh key named and shaped like an upload, which storage
can't tell from one), writes an object framed like Parquet, reads it back and deletes it, and signs a `probe`
record of it. A failed probe, or a look at a completed upload storage did not answer or answered with a server
error, is reported to the API at most once a minute (`POST /v1/storage/witness` with the signed record).

The API logs each report as a `witness` entry at the block it arrived. Wherever
`TASK_API_STORAGE_WITNESS_QUORUM` (3) distinct validators reported storage failing within
`TASK_API_STORAGE_WITNESS_WINDOW_S` (300 s) of each other, storage was down for miners, whatever the API's own
probes saw, and a claim alive then lapses at no cost (`storage_witnessed_outage`). The span is a function of the
logged witnesses, so a replay forgives the same lapses, and an API that logs the witnesses but charges the lapse
diverges from its replay. An API that does not log a validator's report is caught as for a miner's request: the
validator keeps a witness of its own report, and the log owes it an entry (`unlogged`).

## Upload log chain

Each upload log file names the SHA-256 of the previous file's bytes (`prev`) and when it was written
(`prev_written_at`); `log/uploads/latest.json`, signed too, names the newest file's hash. A retry writes the same
bytes, and a copy that lost its Redis or took the log over goes on from the newest published file. A validator
checks every file follows the one it read before it (a fork is fraud), and when a file is missing below the newest
it waits two minutes, then reads on: if the next file shows the missing one was written within the day the bucket
keeps files, it was withheld, and the validator keeps a signed fraud proof of it (the API's own next file names
it) with its witness proofs; older, it was let go. A joining validator checks the links of the files it reads back.

## Storage canary

Every `TASK_API_CANARY_EVERY_S` seconds (60) the API writes an object through a presigned link, as a miner
does, reads it back through another, as a validator does, and deletes it. The object is named and shaped like a
miner's upload (a random key, Parquet's magic bytes, a random size of 16 to 256 KB), so storage can't serve the
canary and fail miners. Each probe is kept a week and public at `GET /v1/storage/probes` with the outages they
show (from the last probe that passed to the next one that does).

A failed probe opens a storage outage under block A's rule ([Rules](rules.md#outages)), as a failed write of the
API's own storage probe does: a claim that expires or is abandoned while it was alive costs its miner nothing, and
the reclaim is logged with cause `storage_outage`, like a restart. The outage's start and end are in the log, so a
replay forgives the same claims.

## What depends on other blocks

- **Block B (verifiable log).** Outcome proofs read the anchored log through B's follower (entries answering a
  request, indexed by its hash); arrival proofs read its `requests` entries, and the API's signed receipts while
  no follower runs. Validators' storage witnesses are `witness` entries the replay reads.
- **Block D (TEE).** Until TLS ends inside the API's enclave, the operator's network in front of the API reads a
  relayed request's `X-Hotkey` like any other; relays then stop a block on the miner's own route, and the
  witnesses make a block of relayed requests provable, but only D makes a relay's traffic indistinguishable.
  Block heights in witnesses come from each validator's own node; D's light client gives the API the same.
