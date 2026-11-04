#!/usr/bin/env bash
# Rebuilds the task API image from a tag or commit exactly as CI does and prints its digest, so anyone can
# check that the image the TEE runs (its digest is in the measured compose) was built from that source.
#
#   task-api/deploy/rebuild.sh task-api-v1.2.0                  # linux/amd64, as CI builds it
#   task-api/deploy/rebuild.sh task-api-v1.2.0 linux/arm64      # another platform gives another digest
#   TASK_API_SOURCE=. task-api/deploy/rebuild.sh HEAD           # a local checkout instead of a fresh clone
#
# Needs only git and docker: BuildKit runs in a container, pinned to the version CI uses.
set -euo pipefail

ref=${1:?usage: rebuild.sh <tag-or-commit> [platform]}
platform=${2:-linux/amd64}
repo=${TASK_API_REPO:-https://github.com/Desearch-ai/subnet-22.git}
buildkit=moby/buildkit:v0.34.1@sha256:99368f3e46177c6405b9740e4461d91cee004128daeecca78d4ceac50ec1da5c

# Under the home directory by default, which Docker Desktop and colima share with their VM; /tmp may not be.
parent=${TASK_API_WORK:-${XDG_CACHE_HOME:-$HOME/.cache}}
mkdir -p "$parent"
work=$(mktemp -d "$parent/sn22-rebuild.XXXXXX")
trap 'rm -rf "$work"' EXIT
if [[ -n "${TASK_API_SOURCE:-}" ]]; then
    # A clean copy of the commit, so files not in git cannot change the result.
    git -C "$TASK_API_SOURCE" archive --format=tar --prefix=src/ "$ref" | tar -x -C "$work"
    commit=$(git -C "$TASK_API_SOURCE" rev-parse "$ref^{commit}")
    epoch=$(git -C "$TASK_API_SOURCE" log -1 --format=%ct "$commit")
else
    git clone --quiet "$repo" "$work/src"
    git -C "$work/src" checkout --quiet --detach "$ref"
    commit=$(git -C "$work/src" rev-parse HEAD)
    epoch=$(git -C "$work/src" log -1 --format=%ct HEAD)
fi
mkdir -p "$work/out"

echo "building $commit ($ref) for $platform, SOURCE_DATE_EPOCH=$epoch" >&2
docker run --rm --privileged \
    -v "$work/src:/context:ro" -v "$work/out:/out" \
    --entrypoint buildctl-daemonless.sh "$buildkit" \
    build \
    --frontend dockerfile.v0 \
    --local context=/context \
    --local dockerfile=/context/task-api \
    --opt filename=Dockerfile.rust \
    --opt platform="$platform" \
    --opt build-arg:SOURCE_DATE_EPOCH="$epoch" \
    --opt build-arg:TASK_API_COMMIT="$commit" \
    --output type=oci,dest=/out/image.tar,rewrite-timestamp=true,oci-mediatypes=true,compression=gzip,compression-level=6,force-compression=true \
    --metadata-file /out/metadata.json >&2

# The image manifest the OCI layout names, the digest CI pushes and the compose pins.
digest=$(tar -xOf "$work/out/image.tar" index.json | sed -n 's/.*"digest":"\(sha256:[0-9a-f]\{64\}\)".*/\1/p' | head -1)
[[ -n "$digest" ]] || { echo "no image manifest in the build's index.json" >&2; exit 1; }
if [[ -n "${TASK_API_IMAGE_OUT:-}" ]]; then
    cp "$work/out/image.tar" "$TASK_API_IMAGE_OUT"
fi
echo "$digest"
