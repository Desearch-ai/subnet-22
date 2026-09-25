//! The shared domain list in Postgres, as one crawler process sees and updates it.

use std::collections::HashSet;
use std::sync::Arc;

use anyhow::{Context, Result};
use chrono::{DateTime, SecondsFormat, Utc};
use indexmap::IndexMap;
use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::crypto::{verify_tls12_signature, verify_tls13_signature, CryptoProvider};
use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
use rustls::{DigitallySignedStruct, SignatureScheme};
use serde_json::json;
use tokio_postgres::Client;

use crate::buckets::{bucket_of, Buckets};
use crate::crawl::{adopted, DomainWrite};
use crate::states::Outcome;

const SYNCED: &str = "synced";
const PAGE: i64 = 20_000;

#[derive(Clone, Debug)]
pub struct Change {
    pub host: String,
    pub rank: Option<i64>,
    pub tld_group: Option<String>,
    pub state: String,
    pub state_reason: Option<String>,
    pub categories: Option<Vec<String>>,
}

/// What Postgres hears about one finished visit.
#[derive(Clone, Debug)]
pub struct Report {
    pub host: String,
    pub state: &'static str,
    pub reason: Option<String>,
    pub checked_at: i64,
    pub urls: i64,
    pub canonical: Option<String>,
    pub redirected: bool,
}

impl Report {
    pub fn of(write: &DomainWrite, urls: i64) -> Self {
        Report {
            host: write.host.clone(),
            state: write.state.as_str(),
            reason: write.reason.clone(),
            checked_at: write.checked_at,
            urls,
            canonical: write.canonical_host.clone(),
            redirected: write.visit.outcome == Outcome::Redirect,
        }
    }
}

/// Reports visits to Postgres in bulk and brings back what changed in this process's buckets.
pub struct Registry {
    dsn: Option<String>,
    client: Option<Client>,
    since: Option<(DateTime<Utc>, String)>,
    buckets: Vec<i16>,
}

impl Registry {
    /// A registry that hears nothing and says nothing, for trials on copies of the stores.
    pub fn offline() -> Self {
        Registry { dsn: None, client: None, since: None, buckets: Vec::new() }
    }

    pub fn new(dsn: String, buckets: &Buckets) -> Result<Self> {
        let mut marks = Vec::new();
        for store in buckets.stores() {
            if let Some(mark) = store.meta(SYNCED)? {
                let at = mark.get(0).and_then(|v| v.as_str()).map(str::to_string);
                let host = mark.get(1).and_then(|v| v.as_str()).map(str::to_string);
                if let (Some(at), Some(host)) = (at, host) {
                    marks.push((at, host));
                }
            }
        }
        let since = match marks.into_iter().min() {
            Some((at, host)) => Some((DateTime::parse_from_rfc3339(&at).context("sync mark")?.with_timezone(&Utc), host)),
            None => None,
        };
        let buckets = buckets.owned().into_iter().map(|b| b as i16).collect();
        Ok(Registry { dsn: Some(dsn), client: None, since, buckets })
    }

    async fn client(&mut self) -> Result<Option<&Client>> {
        let Some(dsn) = &self.dsn else {
            return Ok(None);
        };
        if self.client.as_ref().is_none_or(Client::is_closed) {
            self.client = Some(connect(dsn).await?);
        }
        Ok(self.client.as_ref())
    }

    pub async fn report(&mut self, visits: &[Report], now: i64) -> Result<()> {
        let Some(client) = self.client().await? else {
            return Ok(());
        };
        let latest: IndexMap<&str, &Report> = visits.iter().map(|r| (r.host.as_str(), r)).collect();
        if !latest.is_empty() {
            let rows: Vec<&Report> = latest.into_values().collect();
            let hosts: Vec<&str> = rows.iter().map(|r| r.host.as_str()).collect();
            let states: Vec<&str> = rows.iter().map(|r| r.state).collect();
            let reasons: Vec<Option<&str>> = rows.iter().map(|r| r.reason.as_deref()).collect();
            let checked: Vec<DateTime<Utc>> = rows.iter().map(|r| micros_to_time(r.checked_at)).collect();
            let urls: Vec<i64> = rows.iter().map(|r| r.urls).collect();
            let canonical: Vec<Option<&str>> = rows.iter().map(|r| r.canonical.as_deref()).collect();
            client
                .execute(
                    "UPDATE bot.domains d SET
                        state = s.state, state_reason = s.reason, checked_at = s.checked_at,
                        url_count = s.urls, canonical_host = s.canonical
                     FROM unnest($1::text[], $2::text[], $3::text[], $4::timestamptz[], $5::bigint[], $6::text[])
                        AS s(host, state, reason, checked_at, urls, canonical)
                     WHERE d.host = s.host AND d.state <> 'excluded'",
                    &[&hosts, &states, &reasons, &checked, &urls, &canonical],
                )
                .await?;
        }
        let targets: Vec<(String, &str, &str, &str, Option<&str>)> = visits
            .iter()
            .filter(|r| r.redirected)
            .filter_map(|r| {
                let target = r.canonical.clone()?;
                let (group, state, reason) = adopted(&target);
                Some((target, r.host.as_str(), group, state.as_str(), reason))
            })
            .collect();
        if !targets.is_empty() {
            let hosts: Vec<&str> = targets.iter().map(|t| t.0.as_str()).collect();
            let sources: Vec<&str> = targets.iter().map(|t| t.1).collect();
            let groups: Vec<&str> = targets.iter().map(|t| t.2).collect();
            let states: Vec<&str> = targets.iter().map(|t| t.3).collect();
            let reasons: Vec<Option<&str>> = targets.iter().map(|t| t.4).collect();
            let buckets: Vec<i16> = hosts.iter().map(|h| bucket_of(h) as i16).collect();
            client
                .execute(
                    "INSERT INTO bot.domains (host, rank, tld_group, state, state_reason, bucket, changed_at)
                     SELECT DISTINCT ON (t.host) t.host, d.rank, t.tld_group, t.state, t.reason, t.bucket, $7
                     FROM unnest($1::text[], $2::text[], $3::text[], $4::text[], $5::text[], $6::smallint[])
                        AS t(host, source, tld_group, state, reason, bucket)
                     LEFT JOIN bot.domains d ON d.host = t.source
                     ORDER BY t.host, d.rank NULLS LAST
                     ON CONFLICT (host) DO NOTHING",
                    &[&hosts, &sources, &groups, &states, &reasons, &buckets, &micros_to_time(now)],
                )
                .await?;
        }
        Ok(())
    }

    /// Domains in these buckets changed centrally since the last look, oldest change first.
    pub async fn changes(&mut self, buckets: &Buckets) -> Result<Vec<Change>> {
        let (after, host) = self.since.clone().unwrap_or((DateTime::UNIX_EPOCH, String::new()));
        let owned = self.buckets.clone();
        let Some(client) = self.client().await? else {
            return Ok(Vec::new());
        };
        let rows = client
            .query(
                "SELECT host, rank, tld_group, state, state_reason, categories, changed_at
                 FROM bot.domains
                 WHERE bucket = ANY($1::smallint[]) AND changed_at >= $2
                   AND (changed_at > $2 OR host COLLATE \"C\" > $3)
                 ORDER BY changed_at, host COLLATE \"C\"
                 LIMIT $4",
                &[&owned, &after, &host, &PAGE],
            )
            .await?;
        let Some(last) = rows.last() else {
            return Ok(Vec::new());
        };
        let since: (DateTime<Utc>, String) = (last.get("changed_at"), last.get("host"));
        let format = if since.0.timestamp_subsec_micros() == 0 { SecondsFormat::Secs } else { SecondsFormat::Micros };
        let mark = json!([since.0.to_rfc3339_opts(format, false), since.1]);
        for store in buckets.stores() {
            store.set_meta(SYNCED, &mark)?;
        }
        self.since = Some(since);
        Ok(rows
            .iter()
            .map(|row| Change {
                host: row.get("host"),
                rank: row.get::<_, Option<i32>>("rank").map(i64::from),
                tld_group: row.get("tld_group"),
                state: row.get("state"),
                state_reason: row.get("state_reason"),
                categories: row.get("categories"),
            })
            .collect())
    }
}

pub async fn excluded_categories(dsn: &str) -> Result<HashSet<String>> {
    let client = connect(dsn).await?;
    let rows = client.query("SELECT category FROM bot.excluded_categories", &[]).await?;
    Ok(rows.iter().map(|row| row.get(0)).collect())
}

async fn connect(dsn: &str) -> Result<Client> {
    let tls = tokio_postgres_rustls::MakeRustlsConnect::new(tls_config()?);
    let (client, connection) = tokio_postgres::connect(dsn, tls).await.context("connecting to Postgres")?;
    tokio::spawn(async move {
        if let Err(error) = connection.await {
            eprintln!("postgres connection ended: {error}");
        }
    });
    Ok(client)
}

/// Encrypted, without verifying the server.
fn tls_config() -> Result<rustls::ClientConfig> {
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    Ok(rustls::ClientConfig::builder_with_provider(provider.clone())
        .with_safe_default_protocol_versions()?
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(AnyCertificate(provider)))
        .with_no_client_auth())
}

#[derive(Debug)]
struct AnyCertificate(Arc<CryptoProvider>);

impl ServerCertVerifier for AnyCertificate {
    fn verify_server_cert(
        &self,
        _: &CertificateDer<'_>,
        _: &[CertificateDer<'_>],
        _: &ServerName<'_>,
        _: &[u8],
        _: UnixTime,
    ) -> Result<ServerCertVerified, rustls::Error> {
        Ok(ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        verify_tls12_signature(message, cert, dss, &self.0.signature_verification_algorithms)
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        verify_tls13_signature(message, cert, dss, &self.0.signature_verification_algorithms)
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.0.signature_verification_algorithms.supported_schemes()
    }
}

fn micros_to_time(micros: i64) -> DateTime<Utc> {
    DateTime::from_timestamp_micros(micros).unwrap_or_default()
}
