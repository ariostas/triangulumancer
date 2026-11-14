//! The fixture corpus reader shared by the integration tests.
//!
//! `tests/fixtures/core.json.gz` holds the output of every public method of the
//! 0.3.0 C++ oracle over a corpus of configurations (see
//! `scripts/gen_fixtures.py`). Each test binary includes this module and uses
//! what it needs, hence the `dead_code` allowance.

#![allow(dead_code)]

use std::fs::File;
use std::path::Path;

use flate2::read::GzDecoder;
use serde_json::Value;
use triangulumancer::config::{Config, Kind};

/// A gzipped JSON fixture file under `tests/fixtures`.
pub fn fixture(name: &str) -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    let file = File::open(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_reader(GzDecoder::new(file)).expect("fixture is valid JSON")
}

/// The core corpus.
pub fn corpus() -> Value {
    fixture("core.json.gz")
}

/// A configuration from a JSON array of rows.
pub fn build<K: Kind>(matrix: &[Value]) -> Config<K> {
    let rows: Vec<Vec<i64>> = matrix
        .iter()
        .map(|r| {
            r.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_i64().unwrap())
                .collect()
        })
        .collect();
    let dim = rows.first().map_or(0, Vec::len);
    let flat: Vec<i64> = rows.concat();
    Config::from_row_major(&flat, rows.len(), dim).unwrap()
}

/// A JSON list of simplices, as label lists.
pub fn simplex_list(v: &Value) -> Vec<Vec<usize>> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|s| {
            s.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as usize)
                .collect()
        })
        .collect()
}

/// The methods whose results for `entry` must not be asserted against
/// (`tests/fixtures/KNOWN_DIVERGENT.md`).
pub fn untrusted(entry: &Value) -> Vec<&str> {
    entry["untrusted"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect()
}

/// Every triangulation in a trusted result, whatever the method's output
/// shape: one for the single-triangulation methods, a list for the
/// enumerations and `neighbors_of_placing`, and the `pre`/`post` pairs of
/// `flips_of_placing` (which are partial triangulations, not whole ones).
pub fn triangulations(method: &str, ok: &Value) -> Vec<Vec<Vec<usize>>> {
    let items = ok.as_array().unwrap();
    if method.starts_with("all_") || method == "neighbors_of_placing" {
        items.iter().map(simplex_list).collect()
    } else if method == "flips_of_placing" {
        items
            .iter()
            .flat_map(|f| [simplex_list(&f["pre"]), simplex_list(&f["post"])])
            .collect()
    } else {
        vec![simplex_list(ok)]
    }
}
