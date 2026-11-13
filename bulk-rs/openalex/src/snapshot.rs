//! Works from the OpenAlex works Parquet snapshot, reading only the column chunks a page needs, by HTTP range reads.

use std::sync::Arc;

use anyhow::{anyhow, Result};
use arrow_array::cast::AsArray;
use arrow_array::types::Date32Type;
use arrow_array::{Array, ArrayRef, BooleanArray, ListArray, RecordBatch, StructArray};
use bulk_core::ranged::{column_chunks, int_at, leaves, metadata, RangeClient, Strings};
use parquet::arrow::arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions, ParquetRecordBatchReaderBuilder};
use parquet::arrow::ProjectionMask;
use serde_json::Value;

use crate::work::{inverted_of, Work};

/// The leaf columns a work's checks and page read; `abstract_inverted_index` is a JSON string here.
pub const LEAVES: [&str; 20] = [
    "id",
    "doi",
    "title",
    "publication_year",
    "publication_date",
    "language",
    "type",
    "cited_by_count",
    "is_retracted",
    "is_paratext",
    "abstract_inverted_index",
    "ids",
    "authorships.author.display_name",
    "authorships.institutions.display_name",
    "locations.landing_page_url",
    "locations.pdf_url",
    "primary_location.landing_page_url",
    "primary_location.source.display_name",
    "best_oa_location.license",
    "biblio",
];

fn column<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a ArrayRef> {
    batch.column_by_name(name).ok_or_else(|| anyhow!("the works file has no {name} column"))
}

fn child<'a>(parent: &'a StructArray, name: &str) -> Result<&'a ArrayRef> {
    parent.column_by_name(name).ok_or_else(|| anyhow!("a struct without {name}"))
}

fn structs(array: &ArrayRef) -> Result<&StructArray> {
    array.as_struct_opt().ok_or_else(|| anyhow!("{} rather than a struct", array.data_type()))
}

fn lists(array: &ArrayRef) -> Result<&ListArray> {
    array.as_list_opt::<i32>().ok_or_else(|| anyhow!("{} rather than a list", array.data_type()))
}

fn bools(array: &ArrayRef) -> Result<&BooleanArray> {
    array.as_boolean_opt().ok_or_else(|| anyhow!("{} rather than booleans", array.data_type()))
}

fn owned(strings: &Strings, i: usize) -> String {
    strings.get(i).unwrap_or("").to_string()
}

/// `YYYY-MM-DD` of a day count since 1970-01-01.
pub fn civil(days: i64) -> String {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!("{year:04}-{month:02}-{day:02}")
}

/// The string field `name` of each row's struct in `parent`, empty where the struct or field is null.
struct Field<'a> {
    parent: &'a StructArray,
    values: Strings<'a>,
}

impl<'a> Field<'a> {
    fn of(parent: &'a StructArray, name: &str) -> Result<Self> {
        Ok(Field { parent, values: Strings::of(child(parent, name)?)? })
    }

    fn get(&self, i: usize) -> String {
        if self.parent.is_valid(i) {
            owned(&self.values, i)
        } else {
            String::new()
        }
    }
}

/// The works of one batch that are in the cut, English with at least `min_cited` citations.
pub fn batch_works(batch: &RecordBatch, min_cited: i64) -> Result<Vec<Work>> {
    let language = Strings::of(column(batch, "language")?)?;
    let cited = column(batch, "cited_by_count")?;
    let wanted: Vec<usize> =
        (0..batch.num_rows()).filter(|&i| language.get(i) == Some("en")).filter(|&i| int_at(cited, i).ok().flatten().unwrap_or(0) >= min_cited).collect();
    if wanted.is_empty() {
        return Ok(Vec::new());
    }
    let (id, doi, title, kind) =
        (Strings::of(column(batch, "id")?)?, Strings::of(column(batch, "doi")?)?, Strings::of(column(batch, "title")?)?, Strings::of(column(batch, "type")?)?);
    let inverted = Strings::of(column(batch, "abstract_inverted_index")?)?;
    let year = column(batch, "publication_year")?;
    let date = column(batch, "publication_date")?.as_primitive_opt::<Date32Type>().ok_or_else(|| anyhow!("publication_date is not a date"))?;
    let (retracted, paratext) = (bools(column(batch, "is_retracted")?)?, bools(column(batch, "is_paratext")?)?);
    let ids = column(batch, "ids")?.as_map_opt().ok_or_else(|| anyhow!("ids is not a map"))?;
    let (id_keys, id_values) = (Strings::of(ids.keys())?, Strings::of(ids.values())?);
    let authorships = lists(column(batch, "authorships")?)?;
    let authorship = structs(authorships.values())?;
    let author = structs(child(authorship, "author")?)?;
    let author_name = Field::of(author, "display_name")?;
    let institutions = lists(child(authorship, "institutions")?)?;
    let institution_name = Field::of(structs(institutions.values())?, "display_name")?;
    let locations = lists(column(batch, "locations")?)?;
    let location = structs(locations.values())?;
    let (location_landing, location_pdf) = (Field::of(location, "landing_page_url")?, Field::of(location, "pdf_url")?);
    let primary = structs(column(batch, "primary_location")?)?;
    let primary_landing = Field::of(primary, "landing_page_url")?;
    let primary_source = structs(child(primary, "source")?)?;
    let source_name = Field::of(primary_source, "display_name")?;
    let best_oa = structs(column(batch, "best_oa_location")?)?;
    let license = Field::of(best_oa, "license")?;
    let biblio = structs(column(batch, "biblio")?)?;
    let (volume, issue, first_page, last_page) =
        (Field::of(biblio, "volume")?, Field::of(biblio, "issue")?, Field::of(biblio, "first_page")?, Field::of(biblio, "last_page")?);
    let range = |list: &ListArray, i: usize| if list.is_valid(i) { list.value_offsets()[i] as usize..list.value_offsets()[i + 1] as usize } else { 0..0 };
    let mut works = Vec::with_capacity(wanted.len());
    for i in wanted {
        let (inverted, bad_index) = match inverted.get(i) {
            Some(json) if !json.is_empty() && json != "null" => match serde_json::from_str::<Value>(json) {
                Ok(Value::Object(index)) => match inverted_of(&index) {
                    Ok(inverted) => (Some(inverted), false),
                    Err(_) => (None, true),
                },
                Ok(Value::Null) => (None, false),
                _ => (None, true),
            },
            _ => (None, false),
        };
        let mut pmid = String::new();
        let mut pmcid = String::new();
        if ids.is_valid(i) {
            for j in ids.value_offsets()[i] as usize..ids.value_offsets()[i + 1] as usize {
                match id_keys.get(j) {
                    Some("pmid") => pmid = owned(&id_values, j),
                    Some("pmcid") => pmcid = owned(&id_values, j),
                    _ => {}
                }
            }
        }
        let authors = range(authorships, i)
            .map(|a| {
                let name = if authorship.is_valid(a) && author.is_valid(a) { author_name.get(a) } else { String::new() };
                let held = if authorship.is_valid(a) { range(institutions, a).map(|k| institution_name.get(k)).collect() } else { Vec::new() };
                (name, held)
            })
            .collect();
        let location_urls = range(locations, i).flat_map(|l| [location_landing.get(l), location_pdf.get(l)]).collect();
        let source = if primary.is_valid(i) && primary_source.is_valid(i) { source_name.get(i) } else { String::new() };
        works.push(Work {
            id: owned(&id, i),
            doi: owned(&doi, i),
            title: title.get(i).map(str::to_string),
            year: int_at(year, i)?,
            date: if date.is_valid(i) { civil(i64::from(date.value(i))) } else { String::new() },
            language: "en".into(),
            kind: owned(&kind, i),
            cited: int_at(cited, i)?.unwrap_or(0),
            retracted: retracted.is_valid(i) && retracted.value(i),
            paratext: paratext.is_valid(i) && paratext.value(i),
            inverted,
            bad_index,
            pmid,
            pmcid,
            authors,
            location_urls,
            landing: primary_landing.get(i),
            source,
            oa_license: license.get(i),
            volume: volume.get(i),
            issue: issue.get(i),
            first_page: first_page.get(i),
            last_page: last_page.get(i),
        });
    }
    Ok(works)
}

/// The cut's works in one snapshot file, a row group at a time.
pub fn file_works(client: &RangeClient, url: &str, min_cited: i64) -> Result<Vec<Work>> {
    let (meta, len) = metadata(client, url)?;
    let meta = ArrowReaderMetadata::try_new(Arc::new(meta), ArrowReaderOptions::new())?;
    let schema = meta.metadata().file_metadata().schema_descr();
    let wanted = leaves(schema, &LEAVES);
    let mask = ProjectionMask::leaves(schema, wanted.iter().copied());
    let mut works = Vec::new();
    for group in 0..meta.metadata().num_row_groups() {
        let chunks = column_chunks(client, url, len, meta.metadata().row_group(group), &wanted)?;
        let reader =
            ParquetRecordBatchReaderBuilder::new_with_metadata(chunks, meta.clone()).with_projection(mask.clone()).with_row_groups(vec![group]).build()?;
        for batch in reader {
            works.extend(batch_works(&batch?, min_cited)?);
        }
    }
    Ok(works)
}
