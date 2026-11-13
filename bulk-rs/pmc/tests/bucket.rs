mod support;

use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use bulk_core::http::Client;
use bulk_pmc::bucket::{sorted_names, taken_pmids};
use bulk_pmc::{bounded_map, from_bucket, from_folder, latest_from_inventory, latest_inventory, latest_versions, left_out};
use serde_json::json;
use support::{serve, Reply};

fn client() -> Arc<Client> {
    Arc::new(Client::new("test").unwrap())
}

#[test]
fn pmc_takes_only_open_commercial_unretracted_versions_and_the_newest_of_each() {
    assert_eq!(left_out(&json!({"is_pmc_openaccess": true, "license_code": "CC BY-SA"})), None);
    assert_eq!(left_out(&json!({"is_pmc_openaccess": true, "license_code": "CC BY-NC"})), Some("license"));
    assert_eq!(left_out(&json!({"is_pmc_openaccess": true, "license_code": null})), Some("license"));
    assert_eq!(left_out(&json!({"is_pmc_openaccess": false, "license_code": "CC BY"})), Some("not_open_access"));
    assert_eq!(left_out(&json!({"is_pmc_openaccess": true, "license_code": "CC0", "is_retracted": true})), Some("retracted"));
    assert_eq!(left_out(&json!({"pmcid": "PMC1", "gone": true})), Some("gone"));
    let keys = ["metadata/PMC1.1.json", "metadata/PMC1.3.json", "metadata/PMC1.2.json", "metadata/PMC2.1.json", "README.txt"];
    let newest = latest_versions(keys);
    assert_eq!(newest.into_iter().collect::<Vec<_>>(), [("PMC1".into(), "PMC1.3".into()), ("PMC2".into(), "PMC2.1".into())]);
    let versions = latest_versions(["metadata/PMC5.9.json", "metadata/PMC5.10.json", "metadata/PMC6.x.json", "metadata/XYZ1.1.json"]);
    assert_eq!(versions.get("PMC5").map(String::as_str), Some("PMC5.10"), "versions compare as numbers");
    assert_eq!(versions.len(), 1);
}

#[test]
fn pmc_bucket_reads_keep_only_a_few_articles_ahead_of_the_reader() {
    let pulled = Arc::new(AtomicUsize::new(0));
    let counter = pulled.clone();
    let names = (0..1000u64).inspect(move |_| {
        counter.fetch_add(1, Ordering::SeqCst);
    });
    let mut found = bounded_map(names, 4, 8, |n| n * 2);
    assert_eq!(found.next(), Some(0));
    assert!(pulled.load(Ordering::SeqCst) <= 9, "a reader that waits holds 8 reads in flight, not every name");
    assert_eq!(found.collect::<Vec<_>>(), (1..1000).map(|n| n * 2).collect::<Vec<_>>(), "results come in order");
}

#[test]
fn pmc_bucket_reads_retry_server_errors_and_skip_versions_gone_since_the_inventory() {
    let calls = Arc::new(AtomicUsize::new(0));
    let seen = calls.clone();
    let server = serve(move |request| {
        if request.path.contains("PMC2.1") {
            return Reply::status(404);
        }
        if request.path.ends_with("PMC1.1.json") && seen.fetch_add(1, Ordering::SeqCst) < 2 {
            return Reply::status(500);
        }
        if request.path.ends_with(".json") {
            Reply::ok(r#"{"pmcid": "PMC1", "is_pmc_openaccess": true, "license_code": "CC BY"}"#)
        } else {
            Reply::ok("<article/>")
        }
    });
    let names = ["PMC1.1", "PMC2.1"].map(String::from).into_iter();
    let found: Vec<_> = from_bucket(names, client(), server.base.clone(), 1).map(Result::unwrap).collect();
    assert_eq!(found[0], (json!({"pmcid": "PMC1", "is_pmc_openaccess": true, "license_code": "CC BY"}), Some(b"<article/>".to_vec())));
    assert_eq!(server.count("/metadata/PMC1.1.json"), 3, "two server errors, then the read");
    assert_eq!(left_out(&found[1].0), Some("gone"));
    assert_eq!(found[1], (json!({"pmcid": "PMC2", "gone": true}), None));
}

#[test]
fn pmc_bucket_reads_come_in_name_order_and_read_xml_only_for_taken_versions() {
    let server = serve(|request| {
        let name = request.path.rsplit('/').next().unwrap().trim_end_matches(".json").trim_end_matches(".xml").to_string();
        let n: u64 = name.trim_start_matches("PMC").split('.').next().unwrap().parse().unwrap();
        std::thread::sleep(Duration::from_millis((20 - n) * 5));
        if request.path.ends_with(".json") {
            let licence = if n.is_multiple_of(3) { "CC BY-NC" } else { "CC BY" };
            Reply::ok(format!(r#"{{"pmcid": "PMC{n}", "is_pmc_openaccess": true, "license_code": "{licence}"}}"#))
        } else {
            Reply::ok(format!("<article>{name}</article>"))
        }
    });
    let names: Vec<String> = (1..20).map(|n| format!("PMC{n}.1")).collect();
    let found: Vec<_> = from_bucket(names.clone().into_iter(), client(), server.base.clone(), 8).map(Result::unwrap).collect();
    for (n, (meta, xml)) in (1..20u64).zip(&found) {
        assert_eq!(meta["pmcid"], format!("PMC{n}"));
        let want = (n % 3 != 0).then(|| format!("<article>PMC{n}.1</article>").into_bytes());
        assert_eq!(xml, &want);
    }
    let xml_reads = server.requests().iter().filter(|p| p.ends_with(".xml")).count();
    assert_eq!(xml_reads, (1..20).filter(|n| n % 3 != 0).count(), "left-out versions are not read");
}

#[test]
fn pmc_bucket_reads_hold_at_most_four_names_per_worker_in_flight() {
    let server =
        serve(|request| if request.path.ends_with(".json") { Reply::ok(r#"{"pmcid": "PMC1", "is_pmc_openaccess": false}"#) } else { Reply::status(500) });
    let names: Vec<String> = (1..=100).map(|n| format!("PMC{n}.1")).collect();
    let mut found = from_bucket(names.into_iter(), client(), server.base.clone(), 2);
    assert!(found.next().unwrap().is_ok());
    std::thread::sleep(Duration::from_millis(500));
    assert!(server.requests().len() <= 8, "2 workers keep 8 names in flight, saw {}", server.requests().len());
    assert_eq!(found.count(), 99);
    assert_eq!(server.requests().len(), 100);
}

#[test]
fn pmc_bucket_errors_other_than_404_stop_the_run() {
    let server = serve(|_| Reply::status(403));
    let mut found = from_bucket(["PMC1.1".to_string()].into_iter(), client(), server.base.clone(), 1);
    assert!(found.next().unwrap().is_err());
}

fn inventory_row(key: &str) -> String {
    format!("\"pmc-oa-opendata\",\"{key}\",\"123\",\"2026-11-01T00:00:00.000Z\"\n")
}

#[test]
fn pmc_inventory_csv_files_give_the_newest_version_names_sorted() {
    let dir = tempfile::tempdir().unwrap();
    let plain = dir.path().join("a.csv");
    std::fs::write(&plain, [inventory_row("metadata/PMC9.2.json"), inventory_row("PMC9.2/PMC9.2.xml"), inventory_row("metadata/PMC10.1.json")].concat())
        .unwrap();
    let gz = dir.path().join("b.csv.gz");
    let mut out = flate2::write::GzEncoder::new(std::fs::File::create(&gz).unwrap(), flate2::Compression::default());
    out.write_all([inventory_row("metadata/PMC9.1.json"), inventory_row("metadata/PMC9.3.json"), "\"x\"\n".into()].concat().as_bytes()).unwrap();
    out.finish().unwrap();
    let paths = [plain, gz];
    let newest = latest_from_inventory(&paths).unwrap();
    assert_eq!(newest.into_iter().collect::<Vec<_>>(), [("PMC9".into(), "PMC9.3".into()), ("PMC10".into(), "PMC10.1".into())]);
    assert_eq!(sorted_names(&paths).unwrap(), ["PMC10.1", "PMC9.3"], "names sort as strings");
}

#[test]
fn pmc_inventory_downloads_the_newest_daily_inventory() {
    let listing = r#"<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><Name>pmc-oa-opendata</Name><Prefix>inventory-reports/pmc-oa-opendata/metadata/</Prefix>
<CommonPrefixes><Prefix>inventory-reports/pmc-oa-opendata/metadata/2026-11-09T01-00Z/</Prefix></CommonPrefixes>
<CommonPrefixes><Prefix>inventory-reports/pmc-oa-opendata/metadata/2026-11-10T01-00Z/</Prefix></CommonPrefixes>
<CommonPrefixes><Prefix>inventory-reports/pmc-oa-opendata/metadata/data/</Prefix></CommonPrefixes></ListBucketResult>"#;
    let server = serve(move |request| match request.path.as_str() {
        "/?list-type=2&delimiter=/&prefix=inventory-reports/pmc-oa-opendata/metadata/" => Reply::ok(listing),
        "/inventory-reports/pmc-oa-opendata/metadata/2026-11-10T01-00Z/manifest.json" => Reply::ok(
            r#"{"files": [{"key": "inventory-reports/pmc-oa-opendata/metadata/data/one.csv.gz"}, {"key": "inventory-reports/pmc-oa-opendata/metadata/data/two.csv.gz"}]}"#,
        ),
        path if path.ends_with(".csv.gz") => Reply::ok(path.as_bytes()),
        _ => Reply::status(404),
    });
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("inventory");
    std::fs::create_dir(&dest).unwrap();
    std::fs::write(dest.join("two.csv.gz"), "kept").unwrap();
    let paths = latest_inventory(&dest, &client(), &server.base).unwrap();
    assert_eq!(paths, [dest.join("one.csv.gz"), dest.join("two.csv.gz")]);
    assert_eq!(std::fs::read_to_string(&paths[0]).unwrap(), "/inventory-reports/pmc-oa-opendata/metadata/data/one.csv.gz");
    assert_eq!(std::fs::read_to_string(&paths[1]).unwrap(), "kept", "files already there are not read again");
}

#[test]
fn pmc_pmids_are_those_of_taken_versions_in_name_order() {
    let server = serve(|request| {
        let body = match request.path.as_str() {
            "/metadata/PMC1.1.json" => r#"{"is_pmc_openaccess": true, "license_code": "CC BY", "pmid": 123}"#,
            "/metadata/PMC2.1.json" => r#"{"is_pmc_openaccess": true, "license_code": "CC BY-NC", "pmid": 222}"#,
            "/metadata/PMC3.1.json" => r#"{"is_pmc_openaccess": true, "license_code": "CC0", "pmid": null}"#,
            "/metadata/PMC4.1.json" => r#"{"is_pmc_openaccess": true, "license_code": "CC BY-ND", "pmid": "456"}"#,
            _ => return Reply::status(404),
        };
        Reply::ok(body)
    });
    let names = (1..=4).map(|n| format!("PMC{n}.1"));
    let pmids: Vec<String> = taken_pmids(names, client(), server.base.clone(), 3).map(Result::unwrap).collect();
    assert_eq!(pmids, ["123", "456"]);
    let gone = taken_pmids(["PMC9.1".to_string()].into_iter(), client(), server.base.clone(), 1).collect::<Vec<_>>();
    assert!(gone.is_empty(), "a version gone since the inventory is not taken");
}

#[test]
fn pmc_folders_give_their_json_files_in_name_order_with_xml_for_taken_ones() {
    let dir = tempfile::tempdir().unwrap();
    let write = |name: &str, text: &str| std::fs::write(dir.path().join(name), text).unwrap();
    write("PMC2.1.json", r#"{"pmcid": "PMC2", "is_pmc_openaccess": true, "license_code": "CC BY"}"#);
    write("PMC2.1.xml", "<article/>");
    write("PMC10.1.json", r#"{"pmcid": "PMC10", "is_pmc_openaccess": true, "license_code": "CC BY-NC"}"#);
    write("PMC10.1.xml", "<article/>");
    write("PMC3.1.json", r#"{"pmcid": "PMC3", "is_pmc_openaccess": true, "license_code": "CC BY"}"#);
    write("other.json", "{}");
    let found: Vec<_> = from_folder(dir.path()).unwrap().map(Result::unwrap).collect();
    let pmcids: Vec<_> = found.iter().map(|(meta, _)| meta["pmcid"].as_str().unwrap()).collect();
    assert_eq!(pmcids, ["PMC10", "PMC2", "PMC3"]);
    assert_eq!(found.iter().map(|(_, xml)| xml.is_some()).collect::<Vec<_>>(), [false, true, false]);
}
