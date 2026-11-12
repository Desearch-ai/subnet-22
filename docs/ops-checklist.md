# Operations checklist: what only the team can do

Every step here needs an account, money, a key or a release, so none of it is scripted or done by the
code. Do them in this order; each says what it needs and how to check it. What each protection proves is
in [docs/tee.md](./tee.md); the dstack runbook is [task-api/deploy/dstack/README.md](../task-api/deploy/dstack/README.md).

The commands below use these names; replace them with yours:

```bash
TAG=task-api-v1.0.0                       # the release tag
BUCKET=subnet-22-v2                       # the uploads bucket with Object Lock (the compose names this one)
ENDPOINT=https://s3.us-east-va.io.cloud.ovh.us
```

## 1. An uploads bucket with Object Lock

Needs: the storage account (OVHcloud Object Storage today) and its keys.

The measured compose locks every pinned upload for a day (`TASK_API_UPLOAD_LOCK_S: "86400"`), which only a
versioned bucket created with Object Lock can do. Most stores turn Object Lock on only at creation, so this is
a new bucket:

```bash
aws s3api create-bucket --bucket "$BUCKET" --object-lock-enabled-for-bucket --endpoint-url "$ENDPOINT"
aws s3api get-bucket-versioning --bucket "$BUCKET" --endpoint-url "$ENDPOINT"        # Status: Enabled
aws s3api get-object-lock-configuration --bucket "$BUCKET" --endpoint-url "$ENDPOINT" # ObjectLockEnabled: Enabled, no default rule
```

Then the lifecycle and the public read policy, exactly as in
[docs/witness-relays.md](./witness-relays.md#bucket-configuration). Check that a locked version cannot be
deleted, on a throwaway key:

```bash
echo probe > /tmp/probe && aws s3api put-object --bucket "$BUCKET" --key _test/lock --body /tmp/probe --endpoint-url "$ENDPOINT"
VERSION=$(aws s3api list-object-versions --bucket "$BUCKET" --prefix _test/lock --endpoint-url "$ENDPOINT" --query 'Versions[0].VersionId' --output text)
aws s3api put-object-retention --bucket "$BUCKET" --key _test/lock --version-id "$VERSION" --endpoint-url "$ENDPOINT" \
  --retention "{\"Mode\":\"COMPLIANCE\",\"RetainUntilDate\":\"$(date -u -v+1H +%Y-%m-%dT%H:%M:%SZ)\"}"
aws s3api delete-object --bucket "$BUCKET" --key _test/lock --version-id "$VERSION" --endpoint-url "$ENDPOINT"   # must be refused
```

(`date -v+1H` is macOS; on Linux, `date -u -d '+1 hour' +%Y-%m-%dT%H:%M:%SZ`. The probe stays for that hour,
which is the point.) Make a key pair for the API scoped to this bucket: it becomes `CF_R2_ACCESS_KEY_ID` and
`CF_R2_SECRET_ACCESS_KEY` in step 5. Validators' `--neuron.storage_url` becomes the bucket's public URL (step 7).

## 2. An anchor hotkey registered on SN22

Needs: a coldkey with TAO for the registration burn, on finney.

The log's head is anchored with `Commitments.set_commitment`, which only a hotkey registered on the subnet
may call. It must be a key of its own: the API refuses to start if `TASK_API_ANCHOR_KEY_URI` is its signing
key or the legacy signer (`TASK_API_LEGACY_SIGNER`), and validators ignore anchors from the signer.

```bash
# btcli 9
btcli wallet new-hotkey --wallet-name <owner coldkey> --hotkey sn22-anchor
btcli subnets register --netuid 22 --wallet-name <owner coldkey> --hotkey sn22-anchor --network finney
btcli wallet overview --wallet-name <owner coldkey> --netuids 22 --network finney   # the anchor hotkey has a uid
```

Keep it from being pruned: a uid with no incentive is deregistered once its immunity period ends and the subnet
is full, and anchoring then stops (the API logs every failed anchor; rounds wait for anchors). Watch its uid, and
re-register and re-pin (step 7) if it is lost. Its secret URI (the hotkey's mnemonic as `"<words>"`, or `//...`
for a derived key) is the `TASK_API_ANCHOR_KEY_URI` credential of step 5; its ss58 is pinned in step 7. It needs
no balance beyond what registration leaves: commitments carry no fee.

## 3. CI images with provenance

Needs: push access to the repository (to push the tag) and GitHub Actions minutes.

```bash
git tag "$TAG" <commit> && git push origin "$TAG"
gh run watch "$(gh run list --workflow task-api-image.yml --limit 1 --json databaseId -q '.[0].databaseId')"
```

The workflow builds the image twice, fails unless both builds give one digest, pushes it to
`ghcr.io/desearch-ai/sn22-task-api` by digest and attests its build provenance with Sigstore. Check it from
anywhere:

```bash
DIGEST=sha256:<the digest the run printed>
gh attestation verify "oci://ghcr.io/desearch-ai/sn22-task-api@$DIGEST" --repo Desearch-ai/subnet-22
task-api/deploy/rebuild.sh "$TAG"     # rebuilds from the tag and prints the same digest
```

Make the package public in GitHub's package settings once, so validators and the CVM can pull it without a token.

## 4. The compose for the release

Needs: nothing but the repository; the values come from steps 1 to 3.

Fill in the two placeholders in [`docker-compose.yml`](../task-api/deploy/dstack/docker-compose.yml):
`TASK_API_LEGACY_SIGNER` (the ss58 of today's signer, `TASK_API_SIGNER` in [`desearch/trust.py`](../desearch/trust.py))
and `TASK_API_ADMIN_HOTKEYS` (the feeder's hotkey). Check that `CF_R2_BUCKET` is the bucket of step 1. Then:

```bash
python3 task-api/deploy/dstack/app_compose.py --image-digest "$DIGEST"      # prints the compose hash
python3 task-api/deploy/dstack/app_compose.py --check
```

Commit `app-compose.json`; the printed hash is the version validators approve (step 8).

## 5. A Phala account and the CVM

Needs: a Phala Cloud account with billing (or a dstack host of our own), and a wallet on Base for the
`DstackApp` contract (gas). Everything below is in Phala's dashboard or its CLI; see the
[dstack runbook](../task-api/deploy/dstack/README.md#what-only-you-can-do) for what each choice means.

1. Create the app from `app-compose.json` with the **production** OS image (`dstack-<version>`, never
   `dstack-dev-*`) and the vCPU and memory you will keep: both are measured, and validators pin exactly that.
   Record the image version and size in the release notes.
2. Set the encrypted environment: `CF_R2_ACCESS_KEY_ID`, `CF_R2_SECRET_ACCESS_KEY` (step 1),
   `TASK_API_ANCHOR_KEY_URI` (step 2), and `PAGES_S3_*` if the pages bucket is elsewhere. Nothing else can be
   set: `allowed_envs` lists only these.
3. On the app's `DstackApp` contract (owned by a multisig): `setAllowAnyDevice(false)`, `addDevice(<device id>)`
   for each machine allowed, `addComposeHash(<compose hash>)`.
4. DNS for `api-22.desearch.ai`, not proxied: the gateway `CNAME`, `TXT _dstack-app-address.api-22.desearch.ai
   "<app_id>:443"`, and `CAA 0 issue "letsencrypt.org; validationmethods=tls-alpn-01"`, `CAA 0 issuewild ";"`.
5. Start it. It comes up in standby. Check what it attests and runs with:

```bash
curl -s https://api-22.desearch.ai/v1/config | jq '.settings | {TASK_API_CHAIN_QUORUM, TASK_API_ANCHOR_NETWORK, TASK_API_ANCHOR_QUORUM, TASK_API_UPLOAD_LOCK_S, CF_R2_BUCKET}, .secrets_set'
curl -s https://api-22.desearch.ai/v1/custody | jq '{status, signer}'           # standby, its own signer
python3 tools/verify_attestation.py --api https://api-22.desearch.ai --fresh --compose-hash <compose hash>
```

`verify_attestation.py` fails the `os` check until step 6 is released; every other check must pass. If the API
does not start, its log names the setting it refused (a CVM refuses weak chain, anchor or lock settings).

## 6. The attestation pins in a validator release

Needs: a release of the validator code (a commit to `desearch/attestation.py`, reviewed, then step 7).

Read the measurements and the device id from the CVM itself:

```bash
curl -s https://api-22.desearch.ai/v1/attestation > attestation.json
jq -r '.info.tcb_info | fromjson | "mrtd=\(.mrtd)\nrtmr0=\(.rtmr0)\nrtmr1=\(.rtmr1)\nrtmr2=\(.rtmr2)"' attestation.json
jq -r '.info.device_id' attestation.json
jq -r '.info.compose_hash' attestation.json
```

Cross-check the measurements against dstack's published values for that OS image and size (dstack's
`dstack-mr` tool computes them from the image), not only against what the CVM says. Then pin them:

```python
# desearch/attestation.py
PRODUCTION_OS: tuple["OsMeasurement", ...] = (
    OsMeasurement(name="dstack-<version> <vcpus>vcpu <memory>", mrtd="<mrtd>", rtmr0="<rtmr0>", rtmr1="<rtmr1>", rtmr2="<rtmr2>"),
)
ALLOWED_DEVICES: frozenset[str] = frozenset({"<device id>"})
```

The compose hash is not pinned in code: validators approve it on chain (step 8). Then:

```bash
python3 tools/verify_attestation.py --file attestation.json --compose-hash <compose hash>   # every check passes now
python3 -m pytest -q tests/validators/test_attestation.py
```

## 7. The validator upgrade

Needs: a release, and each validator operator to upgrade.

The release carries step 6's pins and, in [`desearch/trust.py`](../desearch/trust.py), the anchor hotkey of
step 2:

```python
TASK_API_ANCHOR_HOTKEY = "<ss58 of the sn22-anchor hotkey>"
```

It must not be `TASK_API_SIGNER`; validators ignore anchors from the signer. Order:

1. Release the validators first, before the API switches to the locked bucket or the CVM: they accept the old
   bucket and signer as before, and add the pins.
2. Each validator upgrades, sets `--neuron.storage_url` to the bucket of step 1's public URL, keeps
   `--neuron.task_api_url`, and should run the replay (`--neuron.replay_bin`, the `task-api-replay` binary built
   from the same tag). Check the log for `Following the attested task API signer` once custody records exist,
   and that no `anchor` problem is reported.
3. Validators read approvals at old blocks from an archive node: `--neuron.archive_network` if their own node
   prunes state.

## 8. Approval, then the handover

Needs: each validator's hotkey for its commitment (no TAO: commitments are free), and the timelock (50,400
blocks, about 7 days).

```bash
python3 tools/approve_version.py --version <compose hash>                           # dry run: prints the commitment
python3 tools/approve_version.py --version <compose hash> --commit --network finney --wallet.name <cold> --wallet.hotkey <hot>
```

Approval needs more than half the independent validators' stake, from at least three coldkeys, held for the
whole timelock; the owner's hotkeys, validators inheriting stake from them, and stake the owner's coldkey
delegated never count. Once it holds, hand over from today's server, which signs as the legacy signer:

```bash
# signed with an admin hotkey, like an enqueue
POST https://<today's API>/v1/admin/handover  {"successor": "https://<app_id>-443s.<gateway domain>"}
```

Watch `GET /v1/custody` on both: the old server retires, the CVM goes active, and validators log the new signer.
Point the `_dstack-app-address` record at the CVM's app id and stop the old server only after validators follow it.

## What not to skip

- **Two operators for the chain.** The compose reads finney through the Opentensor Foundation's endpoint and
  OnFinality's public one, and both must agree. Before a release, check both answer the calls the API makes
  (`chain_getFinalizedHead`, `state_queryStorageAt`, `state_getKeysPaged`, `state_call`); a third operator's
  endpoint lets the quorum ride out one being down. The API refuses a list where one operator holds the quorum.
- **The anchor hotkey's uid** (step 2): if it is pruned, anchoring stops and rounds wait.
- **No pause without a reason.** A pause stops the accounts and forgives the claims alive in it; it is logged
  with the admin's signed request.
