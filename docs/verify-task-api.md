# Verify the task API

Anyone can check what the task API runs, without credentials and without trusting Desearch. Each step
checks the one before it: the source gives the image, the image is in the compose, the compose is in
the TDX quote, the quote binds the key that signs the log, and validators' stake on chain says which
version is in charge. [How it works](./tee.md).

- [1. The image](#1-the-image)
- [2. The compose hash](#2-the-compose-hash)
- [3. The attestation](#3-the-attestation)
- [4. The custody chain and approvals](#4-the-custody-chain-and-approvals)
- [5. The settings](#5-the-settings)

You need a checkout of this repository, Python 3.10 or newer with `pip install -r requirements.txt`,
and for step 1 `docker` and the [GitHub CLI](https://cli.github.com/).

## 1. The image

Every release is a tag `task-api-vX.Y.Z`. CI builds the image from the tag, pushes it to
`ghcr.io/desearch-ai/sn22-task-api` and signs a provenance (SLSA, through Sigstore) saying which
commit and workflow built which digest.

```bash
gh attestation verify oci://ghcr.io/desearch-ai/sn22-task-api@sha256:<digest> --repo Desearch-ai/subnet-22
```

It passes only for an image built by this repository's workflow, and prints the commit.

You do not have to trust GitHub's builders either: the build is reproducible. Rebuild the tag and
compare the digest it prints with the one in the compose:

```bash
task-api/deploy/rebuild.sh task-api-vX.Y.Z
```

It builds for `linux/amd64`, as CI does; another platform gives another digest.

## 2. The compose hash

The compose the CVM boots is [`task-api/deploy/dstack/docker-compose.yml`](../task-api/deploy/dstack/docker-compose.yml),
wrapped in an `app-compose.json`. Its SHA-256 is the compose hash the quote measures.

```bash
python3 task-api/deploy/dstack/app_compose.py --image-digest sha256:<digest>
```

It writes the `app-compose.json` for this checkout and prints its compose hash. Read the compose
itself: every setting is written in it, every image is pinned by digest, and `allowed_envs` lists only
the storage credentials and the anchor hotkey's secret, the only values the deployer may set. Check that it
reads the chain through endpoints of two operators that must agree, anchors the log
(`TASK_API_ANCHOR_NETWORK`) and locks uploads for a day (`TASK_API_UPLOAD_LOCK_S: "86400"`): the API
refuses to start in a CVM without them, and validators refuse the compose.

## 3. The attestation

```bash
python3 tools/verify_attestation.py --api https://api-22.desearch.ai --fresh
```

It fetches `/v1/attestation` with a random nonce, so the quote is made for you now, and checks:

- the quote is signed by Intel's DCAP chain, its TCB is up to date and debug is off;
- the OS measurements are one of the production images pinned in [`desearch/attestation.py`](../desearch/attestation.py);
- the event log replays to RTMR3, and the compose hash in it is that of the `app_compose` returned;
- the compose passes the same policy validators apply: no build, no privileged container, no host
  network, no docker socket, no `pre_launch_script`, only credentials in `allowed_envs`, the
  governance settings validators pin, and the minimums they require (two endpoints agreeing on chain
  reads and anchors, anchoring on, a day's Object Lock, three coldkeys to approve a version);
- the key provider is dstack's KMS, and the quote's report data binds the document with the signer;
- `/v1/config` hashes to the `config_sha256` the quote binds;
- the TLS certificate the API serves has the key the quote binds (`tls_spki_sha256`), so you are
  talking to the CVM itself, not to a proxy in front of it.

It prints the compose hash, which must be the one you computed in step 2, and the signer. Pass
`--compose-hash` and `--signer` to make the tool check them for you. It exits non-zero when any check
fails.

## 4. The custody chain and approvals

```bash
python3 tools/verify_attestation.py --storage-url https://subnet-22.s3.us-east-va.io.cloud.ovh.us \
  --network wss://archive.chain.opentensor.ai:443
```

It reads every custody record from the uploads bucket and checks them as validators do: each record's
signature and link, that more than half of the independent validators' stake approved its version for
7 days at its block, and the new signer's attestation. It prints the signer validators follow now; it
must be the signer from step 3. Without `--network` it checks everything but the approvals.

Validators' approvals are their commitments on netuid 22, `sn22tee:<compose hash>`. To list them:

```bash
python3 -c "
import asyncio, bittensor as bt
async def main():
    async with bt.Client('finney') as chain:
        for row in await chain.read('commitments', netuid=22):
            print(row['hotkey'], row['block'], row['commitment'])
asyncio.run(main())"
```

## 5. The settings

```bash
curl -s https://api-22.desearch.ai/v1/config | python3 -m json.tool
```

Every setting the running API uses, by its environment variable. Compare them with the compose from
step 2: they must be the same, and step 3 already checked that this answer is the one the quote binds.
