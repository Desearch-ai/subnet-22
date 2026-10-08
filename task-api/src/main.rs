//! The SN22 task API: hands out crawl and embed tasks, takes uploads and verdicts, keeps the accounts.

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

use std::net::SocketAddr;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use tokio::net::TcpListener;

use task_api::settings::{RegistryMode, Settings};
use task_api::state::{redis_from, State};
use task_api::{http, janitor};

/// Docker marks the copy unhealthy after three failed 5 s health checks, and the proxy then stops routing to it.
const DRAIN: Duration = Duration::from_secs(18);

/// True when asked to terminate, as a deploy does, rather than interrupted by hand.
async fn shutdown() -> bool {
    let interrupt = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        if let Ok(mut signal) = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            signal.recv().await;
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        _ = interrupt => false,
        _ = terminate => true,
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let settings = Settings::from_env()?;
    let listen = settings.listen.clone();
    let (storage, pages) = State::buckets(&settings)?;
    for bucket in [&storage, &pages] {
        bucket.check().await.with_context(|| format!("R2 bucket {} is not reachable", bucket.bucket))?;
    }
    let redis = redis_from(&settings.redis_url).await?;
    let state = Arc::new(State::new(settings, redis, storage, pages).await?);
    if let (Some(chain), RegistryMode::Chain { netuid, .. }) = (&state.chain, &state.settings.registry) {
        state.registry.refresh(chain.clone(), *netuid);
    }
    let (stop, stopped) = tokio::sync::watch::channel(false);
    let janitor = tokio::spawn(janitor::run(state.clone(), stopped));
    let listener = TcpListener::bind(&listen).await.with_context(|| format!("listening on {listen}"))?;
    eprintln!("task API on {listen}, signing as {}", state.signer());
    let stopping = {
        let state = state.clone();
        async move {
            let terminated = shutdown().await;
            let _ = stop.send(true);
            if terminated {
                state.draining.store(true, Ordering::Relaxed);
                tokio::time::sleep(DRAIN).await;
            }
        }
    };
    axum::serve(listener, http::router(state).into_make_service_with_connect_info::<SocketAddr>()).with_graceful_shutdown(stopping).await?;
    janitor.await?;
    Ok(())
}
