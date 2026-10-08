# Build from this folder: docker build -t desearch-publisher publisher/
FROM rust:1-bookworm AS build

RUN apt-get update \
    && apt-get install -y --no-install-recommends clang libclang-dev cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src src
COPY bench bench
ARG CARGO_BUILD_JOBS=4
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --release --locked --bin publisher \
    && strip -o /publisher target/release/publisher

FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates tini \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 publisher \
    && mkdir -p /data \
    && chown publisher /data

COPY --from=build /publisher /usr/local/bin/publisher
COPY healthcheck.sh /usr/local/bin/publisher-healthcheck
RUN chmod 755 /usr/local/bin/publisher-healthcheck

USER publisher
ENV PUBLISHER_INDEX=/data/index \
    PUBLISHER_HEARTBEAT_PATH=/data/heartbeat \
    PUBLISHER_HEARTBEAT_MAX_AGE_S=600 \
    MALLOC_ARENA_MAX=2
VOLUME /data

# No HEALTHCHECK instruction. debian:bookworm-slim has /bin/sh, sed, date, head, stat, and find,
# and /usr/local/bin/publisher-healthcheck uses those. It is wired in docker-compose.yml with
# `healthcheck.disable: true` until the measured max heartbeat staleness is known. Enabling it
# sooner marks a healthy publisher unhealthy. This image has no HTTP server and no curl or wget.

# tini passes SIGTERM on, so the publisher finishes the batch in hand before it exits.
ENTRYPOINT ["/usr/bin/tini", "--", "publisher"]
CMD ["serve"]
