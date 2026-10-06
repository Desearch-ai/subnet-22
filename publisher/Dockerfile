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
RUN cargo build --release --locked --bin publisher \
    && strip target/release/publisher

FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates tini \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 publisher \
    && mkdir -p /data \
    && chown publisher /data

COPY --from=build /src/target/release/publisher /usr/local/bin/publisher

USER publisher
ENV PUBLISHER_INDEX=/data/index \
    MALLOC_ARENA_MAX=2
VOLUME /data

# tini passes SIGTERM on, so the publisher finishes the batch in hand before it exits.
ENTRYPOINT ["/usr/bin/tini", "--", "publisher"]
CMD ["serve"]
