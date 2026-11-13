mod support;

use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::sync::Arc;

use arrow_array::builder::{Int64Builder, MapBuilder, StringBuilder};
use arrow_array::{ArrayRef, RecordBatch};
use bulk_pmc::order::RangeClient;
use bulk_pmc::{citations, order, pmc_ids};
use indexmap::IndexMap;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use support::{serve, Reply};

const PUBMED: &str = "https://pubmed.ncbi.nlm.nih.gov/";

type Work<'a> = (Vec<(&'a str, String)>, Option<i64>);

fn works(rows: &[Work], row_group_rows: usize) -> Vec<u8> {
    let mut ids = MapBuilder::new(None, StringBuilder::new(), StringBuilder::new());
    let mut cited = Int64Builder::new();
    for (entries, count) in rows {
        for (key, value) in entries {
            ids.keys().append_value(key);
            ids.values().append_value(value);
        }
        ids.append(true).unwrap();
        cited.append_option(*count);
    }
    let mut title = StringBuilder::new();
    for _ in rows {
        title.append_value("a title the reader leaves alone");
    }
    let columns: Vec<(&str, ArrayRef)> =
        vec![("title", Arc::new(title.finish())), ("ids", Arc::new(ids.finish())), ("cited_by_count", Arc::new(cited.finish()))];
    let batch = RecordBatch::try_from_iter(columns).unwrap();
    let mut out = Vec::new();
    let properties = WriterProperties::builder().set_max_row_group_row_count(Some(row_group_rows)).build();
    let mut writer = ArrowWriter::try_new(&mut out, batch.schema(), Some(properties)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    out
}

#[test]
fn pmc_versions_go_most_cited_per_year_first_by_their_pmid_in_openalex() {
    let dir = tempfile::tempdir().unwrap();
    let ids_path = dir.path().join("PMC-ids.csv.gz");
    let mut out = flate2::write::GzEncoder::new(std::fs::File::create(&ids_path).unwrap(), flate2::Compression::default());
    out.write_all(
        b"Journal Title,ISSN,eISSN,Year,Volume,Issue,Page,DOI,PMCID,PMID,Manuscript Id,Release Date\n\
J,,,2006,1,1,1,10.1/a,PMC1,11,,live\n\
J,,,2025,1,1,1,10.1/b,PMC2,22,,live\n\
J,,,2025,1,1,1,10.1/c,PMC3,33,,live\n\
J,,,2024,1,1,1,10.1/d,PMC4,,,live\n\
J,,,2020,1,1,1,10.1/e,PMC9,99,,live\n",
    )
    .unwrap();
    out.finish().unwrap();
    let pmid = |n: u32| format!("{PUBMED}{n}");
    let rows = vec![
        (vec![("openalex", "W1".to_string()), ("pmid", pmid(11))], Some(210)),
        (vec![("openalex", "W2".to_string()), ("pmid", pmid(22))], Some(18)),
        (vec![("openalex", "W3".to_string()), ("doi", "https://doi.org/10.1/c".to_string())], Some(500)),
        (vec![("openalex", "W9".to_string()), ("pmid", pmid(99))], Some(70)),
        (vec![("pmid", pmid(22))], None),
    ];
    let file = Arc::new(works(&rows, 2));
    let served = file.clone();
    let server = serve(move |request| if request.path == "/part_0000.parquet" { Reply::ranged(&served, request.range.as_deref()) } else { Reply::status(404) });

    let ids = pmc_ids(&ids_path).unwrap();
    assert_eq!(ids["PMC2"], ("22".to_string(), 2025));
    assert_eq!(ids["PMC4"], (String::new(), 2024));
    let wanted: HashSet<String> = ["11", "22", "33"].map(String::from).into();
    let client = Arc::new(RangeClient::new("test").unwrap());
    let url = format!("{}/part_0000.parquet", server.base);
    let cited = citations(&[url], client, Arc::new(wanted), 1).unwrap();
    assert_eq!(cited, HashMap::from([("11".to_string(), 210), ("22".to_string(), 18)]), "PMIDs outside the wanted set are not kept");
    assert_eq!(server.requests().len(), 1 + 3 * 3, "the footer, then the ids and cited_by_count chunks of each row group");

    let names: IndexMap<String, String> = [("PMC1", "PMC1.2"), ("PMC2", "PMC2.1"), ("PMC3", "PMC3.1"), ("PMC4", "PMC4.1"), ("PMC5", "PMC5.1")]
        .into_iter()
        .map(|(a, b)| (a.to_string(), b.to_string()))
        .collect();
    assert_eq!(
        order(&names, &ids, &cited, 2026),
        ["PMC1.2", "PMC2.1", "PMC5.1", "PMC4.1", "PMC3.1"],
        "210 citations in 21 years beat 18 in 2; the rest go newest first"
    );
}
