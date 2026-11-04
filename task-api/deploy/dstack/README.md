# Running the task API in a TEE

The runbook for the task API as a [dstack](https://github.com/Dstack-TEE/dstack) app in an Intel TDX
CVM, on Phala Cloud or any dstack host. What it proves and how validators check it is in
[docs/tee.md](../../../docs/tee.md); how anyone checks a release is in
[docs/verify-task-api.md](../../../docs/verify-task-api.md).

| File | |
| --- | --- |
| [`docker-compose.yml`](docker-compose.yml) | the app: `api` (the task API image by digest), `redis` (inside the CVM, a Unix socket only, no port), `tls-init` and `tls` (Caddy, ending TLS inside the CVM); every setting written out |
| [`app_compose.py`](app_compose.py) | wraps it into `app-compose.json` the same way every time, refuses anything unpinned or let in from outside, and prints the **compose hash** |
| `app-compose.json` | written by `app_compose.py` for a release: the exact bytes the CVM boots and measures |
| [`../rebuild.sh`](../rebuild.sh) | rebuilds the image from a tag and prints its digest, as CI does |

## What only you can do

These need an account, money or a key, so none of them is scripted here:

1. **A Phala Cloud account** (or a dstack host of our own) with billing, and the app created on it.
2. **The production OS image**: `dstack-<version>`, never `dstack-dev-<version>` (the dev image has SSH
   and a debug console, and its measurements differ). Choose the vCPU and memory once: they are measured
   into RTMR0, so validators pin the measurements of exactly that image and size. Record the image version
   and size in the release notes.
3. **The app's on-chain KMS settings** (the `DstackApp` contract on Base that Phala's KMS asks before it
   gives the CVM its disk key). Deploying and changing it costs gas, and its owner should be a multisig:
   - `setAllowAnyDevice(false)` and `addDevice(<device id>)` for each machine allowed to run the app. The
     device id is `sha256(PPID)` of the machine's TDX platform, shown in the attestation's `info.device_id`
     and in Phala's dashboard. Validators pin the same list.
   - `addComposeHash(<compose hash>)` for each release. This only lets a CVM boot; validators still take
     a version only once their stake approved it and the timelock passed (docs/tee.md).
4. **The storage credentials** as encrypted environment variables of the app: `CF_R2_ACCESS_KEY_ID`,
   `CF_R2_SECRET_ACCESS_KEY` and, if the pages bucket is on another store, `PAGES_S3_ACCESS_KEY_ID` and
   `PAGES_S3_SECRET_ACCESS_KEY`. They are encrypted to the CVM; nothing else can be passed in, because
   `allowed_envs` lists only these and every other value is written in the compose.
5. **DNS for `api-22.desearch.ai`**, DNS only (not proxied through Cloudflare: TLS must end in the CVM):
   - a `CNAME` (or `A`) to the dstack gateway's address;
   - `TXT _dstack-app-address.api-22.desearch.ai "<app_id>:443"`, so the gateway passes the TLS stream
     through to the app's port 443 instead of ending it;
   - `CAA 0 issue "letsencrypt.org; validationmethods=tls-alpn-01"` and `CAA 0 issuewild ";"`, so only
     Let's Encrypt can issue, and only to whoever answers TLS on the domain. Once the CVM has its first
     certificate, add `accounturi=<the CVM's ACME account URL>` to pin issuance to the key Caddy keeps on
     the CVM's encrypted disk. Monitor Certificate Transparency for the domain anyway: clients compare the
     served certificate's key with `tls_spki_sha256` in the attestation, which DNS cannot fake.
6. **The admin hotkeys** (the bot's feeder key) written into `TASK_API_ADMIN_HOTKEYS` in the compose. An
   admin can enqueue URLs, ask for a handover and pause; nothing else.

Nothing here touches mainnet: approvals are commitments each validator makes with its own hotkey.

## A release

1. Tag the commit `task-api-vX.Y.Z` and push the tag. CI
   ([task-api-image.yml](../../../.github/workflows/task-api-image.yml)) builds the image twice, pushes it
   to `ghcr.io/desearch-ai/sn22-task-api` only if both builds give the same digest, attests its build
   provenance with Sigstore, and prints the digest.
2. Put the admin hotkeys in `docker-compose.yml` (they change rarely) and build the app compose:

   ```bash
   python3 task-api/deploy/dstack/app_compose.py --image-digest sha256:<digest CI printed>
   ```

   It writes `app-compose.json` and prints the compose hash. Commit `app-compose.json` and put the hash,
   the image digest, the OS image and the CVM size in the release notes.
3. Validators check the release (docs/verify-task-api.md) and approve it with their commitment: the
   running version's hash and the new one, `sn22tee:<running>,<new>`
   ([tools/approve_version.py](../../../tools/approve_version.py)). The subnet owner's stake does not
   count.
4. Deploy a CVM from `app-compose.json`: it starts in standby, serves `/v1/attestation`, `/v1/config` and
   `/v1/custody`, and takes no work. Check that the compose hash it attests is the one printed in step 2:
   if the deploy tool rewrote `app-compose.json`, the hash differs and validators will not take it.
5. After the timelock (50,400 blocks, about 7 days) from when enough stake approved, hand over (below).

## Genesis and recovery

The first CVM, or one replacing a CVM lost for good, has no predecessor to hand over to it. Deploy it,
read its signer from `/v1/attestation`, and ask validators to check that attestation and commit
`sn22tee:<compose hash>@<signer>`. Once that stake has held it through the timelock, the CVM appoints
itself with an `appoint` record and starts serving; validators follow it from then on.

## Handover

With the successor deployed and approved:

```bash
# signed with an admin hotkey, like an enqueue
POST https://api-22.desearch.ai/v1/admin/handover  {"successor": "https://<successor app_id>-443s.<gateway domain>"}
```

The running CVM checks the successor's attestation and its approval itself, stops taking work, sends it
its state encrypted to the successor's attested key, and retires once the successor confirms. Then point
the `_dstack-app-address` TXT record at the successor's app id. Watch `GET /v1/custody` on both. If the
handover fails before the record is published, the running CVM takes up work again; nothing is lost and
there is never a second signer. Stop the old CVM only after validators follow the new signer.

## Pause

The only emergency power, and it changes no code, setting, key or account:

```bash
POST /v1/admin/pause  {"paused": true, "reason": "..."}     # claims get the refusal PAUSED, nobody is penalized
POST /v1/admin/pause  {"paused": false}
```

## Requirements and limits

- **PROXY protocol.** The app compose asks the gateway (`port_policy`) to send a PROXY protocol header on
  port 443, so Caddy, and through `X-Client-Address` the API, see each caller's address for the per-IP
  limits. A gateway that does not send it still works (Caddy takes the connection's own address), but
  every caller then looks like the gateway, and the per-IP read and sign-in limits apply to all at once.
  Check after a deploy that `/v1/health` callers are not all one address.
- **The dstack socket** is mounted into `api` only, which runs as uid 10001. If the host's socket is
  root-only, the API cannot get its quote: check `/v1/attestation` right after the first boot.
- **Logs are public** (`public_logs`), so a log line must never hold a secret. Caddy writes no access log.
- **Redis** listens on a Unix socket in a volume only `redis` and `api` mount, with no TCP port at all,
  so nothing outside the CVM, not even the host, can read or change queues and claims.
- **Disk.** Every volume is on the CVM's disk, which dstack encrypts with a key the KMS gives only to this
  app's CVM. A host can still restore an old copy of that disk; the custody records make that copy
  harmless (docs/tee.md).

## Checking locally

```bash
python3 task-api/deploy/dstack/app_compose.py --check          # pinned images, allowed_envs, ports, mounts
CF_R2_ACCESS_KEY_ID=x CF_R2_SECRET_ACCESS_KEY=x docker compose -f task-api/deploy/dstack/docker-compose.yml config -q
task-api/deploy/rebuild.sh task-api-vX.Y.Z                      # the digest CI printed
```

The attestation flow runs against dstack's simulator, which answers on a local socket with a real quote
whose report data it patches in (so its signature does not verify): build it with
`sdk/simulator/build.sh` in the [dstack repository](https://github.com/Dstack-TEE/dstack) and point
`TASK_API_DSTACK_SOCKET` at its `dstack.sock`.
