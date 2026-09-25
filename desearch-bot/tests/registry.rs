//! The registry against a throwaway PostgreSQL server with the real schema.

use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use desearch_bot::buckets::{bucket_of, Buckets, Resources};
use desearch_bot::registry::{Registry, Report};
use tokio_postgres::NoTls;

struct Server {
    process: Child,
    data: PathBuf,
    dsn: String,
}

impl Drop for Server {
    fn drop(&mut self) {
        // SIGINT is the fast shutdown; SIGTERM would wait for every client to disconnect.
        let _ = Command::new("kill").args(["-INT", &self.process.id().to_string()]).status();
        let _ = self.process.wait();
        let _ = std::fs::remove_dir_all(&self.data);
    }
}

fn binary(name: &str) -> Option<PathBuf> {
    std::env::split_paths(&std::env::var_os("PATH")?).map(|dir| dir.join(name)).find(|path| path.is_file())
}

fn start() -> Option<Server> {
    let (initdb, postgres) = (binary("initdb")?, binary("postgres")?);
    let data = std::env::temp_dir().join(format!("desearch-bot-pg-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&data);
    let initialised = Command::new(initdb)
        .arg("-D")
        .arg(&data)
        .args(["-U", "postgres", "--auth=trust", "-E", "UTF8", "--no-locale"])
        .env("LC_ALL", "C")
        .env("LANG", "C")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .ok()?;
    assert!(initialised.success(), "initdb failed");
    let port = TcpListener::bind("127.0.0.1:0").ok()?.local_addr().ok()?.port();
    let process = Command::new(postgres)
        .arg("-D")
        .arg(&data)
        .args(["-p", &port.to_string(), "-c", "listen_addresses=127.0.0.1", "-c", "unix_socket_directories=", "-c", "fsync=off"])
        .env("LC_ALL", "C")
        .env("LANG", "C")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    Some(Server { process, data, dsn: format!("postgresql://postgres@127.0.0.1:{port}/postgres") })
}

async fn connect(dsn: &str) -> tokio_postgres::Client {
    for _ in 0..100 {
        if let Ok((client, connection)) = tokio_postgres::connect(dsn, NoTls).await {
            tokio::spawn(connection);
            return client;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    panic!("PostgreSQL did not start");
}

fn report(host: &str, state: &'static str, reason: Option<&str>, canonical: Option<&str>, at: i64) -> Report {
    Report {
        host: host.into(),
        state,
        reason: reason.map(Into::into),
        checked_at: at,
        urls: 42,
        canonical: canonical.map(Into::into),
        redirected: canonical.is_some(),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reports_visits_and_takes_back_changes() {
    let Some(server) = start() else {
        eprintln!("skipped: needs a local PostgreSQL install");
        return;
    };
    let db = connect(&server.dsn).await;
    let schema = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("schema.sql")).unwrap();
    db.batch_execute(&schema).await.unwrap();
    let hosts = ["alpha.com", "beta.com", "gamma.com"];
    for (rank, host) in hosts.iter().enumerate() {
        db.execute(
            "INSERT INTO bot.domains (host, bucket, rank, tld_group) VALUES ($1, $2, $3, 'big_generic')",
            &[host, &(bucket_of(host) as i16), &(rank as i32 + 1)],
        )
        .await
        .unwrap();
    }
    let root = std::env::temp_dir().join(format!("desearch-bot-registry-{}", std::process::id()));
    let mut owned: Vec<usize> = hosts.iter().chain(&["landing-site.com"]).map(|h| bucket_of(h)).collect();
    owned.sort_unstable();
    owned.dedup();
    let resources = Resources::new(8 << 20, 8 << 20, owned.len());
    let buckets = Buckets::open(&root, &owned, &resources).unwrap();

    let mut registry = Registry::new(server.dsn.clone(), &buckets).unwrap();
    assert_eq!(registry.changes(&buckets).await.unwrap().len(), 3);
    assert!(registry.changes(&buckets).await.unwrap().is_empty());

    db.execute("UPDATE bot.domains SET state = 'excluded', state_reason = 'ut1_adult', changed_at = now() WHERE host = 'gamma.com'", &[])
        .await
        .unwrap();
    let now = chrono::Utc::now().timestamp_micros();
    let visits = [
        report("alpha.com", "active", None, None, now),
        report("beta.com", "redirects", Some("redirect"), Some("landing-site.com"), now),
        report("gamma.com", "active", None, None, now),
    ];
    registry.report(&visits, now).await.unwrap();

    let alpha = db.query_one("SELECT state, url_count, checked_at IS NOT NULL FROM bot.domains WHERE host = 'alpha.com'", &[]).await.unwrap();
    assert_eq!((alpha.get::<_, String>(0).as_str(), alpha.get::<_, i64>(1), alpha.get::<_, bool>(2)), ("active", 42, true));
    let beta = db.query_one("SELECT canonical_host FROM bot.domains WHERE host = 'beta.com'", &[]).await.unwrap();
    assert_eq!(beta.get::<_, Option<String>>(0).as_deref(), Some("landing-site.com"));
    let gamma = db.query_one("SELECT state FROM bot.domains WHERE host = 'gamma.com'", &[]).await.unwrap();
    assert_eq!(gamma.get::<_, String>(0), "excluded", "an exclusion made meanwhile stands");
    let landing = db.query_one("SELECT rank, bucket, state FROM bot.domains WHERE host = 'landing-site.com'", &[]).await.unwrap();
    assert_eq!(
        (landing.get::<_, Option<i32>>(0), landing.get::<_, i16>(1) as usize, landing.get::<_, String>(2).as_str()),
        (Some(2), bucket_of("landing-site.com"), "new"),
        "a redirect target joins ranked like the domain that pointed at it"
    );

    let changed = registry.changes(&buckets).await.unwrap();
    let seen: Vec<(&str, &str)> = changed.iter().map(|c| (c.host.as_str(), c.state.as_str())).collect();
    assert!(seen.contains(&("gamma.com", "excluded")) && seen.contains(&("landing-site.com", "new")), "{seen:?}");
    let mut reopened = Registry::new(server.dsn.clone(), &buckets).unwrap();
    assert!(reopened.changes(&buckets).await.unwrap().is_empty(), "the sync mark survives a restart");
    drop(buckets);
    std::fs::remove_dir_all(&root).ok();
}
