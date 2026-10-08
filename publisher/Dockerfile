# Build from the repository root: docker build -f publisher/Dockerfile -t desearch-publisher .
FROM rust:1-bookworm AS build

RUN apt-get update \
    && apt-get install -y --no-install-recommends clang libclang-dev cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY crates/desearch crates/desearch
COPY publisher/Cargo.toml publisher/Cargo.lock publisher/
COPY publisher/src publisher/src
COPY publisher/bench publisher/bench
WORKDIR /src/publisher
ARG CARGO_BUILD_JOBS=4
ENV CARGO_TARGET_DIR=/src/target
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --release --locked --bin publisher \
    && strip -o /publisher /src/target/release/publisher

FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates tini \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 publisher \
    && mkdir -p /data \
    && chown publisher /data

COPY --from=build /publisher /usr/local/bin/publisher

USER publisher
ENV PUBLISHER_INDEX=/data/index \
    MALLOC_ARENA_MAX=2
VOLUME /data

# tini passes SIGTERM on, so the publisher finishes the batch in hand before it exits.
ENTRYPOINT ["/usr/bin/tini", "--", "publisher"]
CMD ["serve"]
