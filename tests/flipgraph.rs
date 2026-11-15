//! The flip-graph enumeration (`all_connected_triangulations`) against the C++
//! oracle, the known-correct B1 values, and TOPCOM's own enumerators.
//!
//! * The core corpus, as sets, for every trusted result the oracle finished.
//! * Bug B1 (`tests/fixtures/KNOWN_DIVERGENT.md`): the correct
//!   `only_fine` counts, and — since the flip graph of every one of those
//!   configurations is connected — the same *set* as the oracle's
//!   `all_triangulations(only_fine=True)`, which is unaffected by B1.
//! * `topcom.json.gz`: counts from `points2ntriangs` and `points2nfinetriangs`,
//!   and whole components from `points2triangs` where they are small. The
//!   default run is bounded to components of at most `DEFAULT_LIMIT`
//!   triangulations; the `#[ignore]`d test runs everything (in release mode,
//!   about four minutes for 2.9 million triangulations:
//!   `cargo test --release --test flipgraph -- --ignored`).
//!
//! Every enumerated triangulation of the smaller results is also `check`ed.

mod common;

use std::collections::BTreeSet;

use common::{build, corpus, fixture, simplex_list, untrusted};
use serde_json::Value;
use triangulumancer::Error;
use triangulumancer::config::{Config, Kind, Points, Vectors};
use triangulumancer::om::Chirotope;
use triangulumancer::triang::{self, FlipGraph, Triangulation};

/// Components larger than this are only enumerated by the ignored test.
const DEFAULT_LIMIT: u64 = 500;

/// Triangulations are `check`ed only in results up to this size.
const CHECK_LIMIT: usize = 500;

type Labels = Vec<Vec<usize>>;

fn as_set(ts: &[Triangulation]) -> BTreeSet<Labels> {
    ts.iter().map(Triangulation::to_vecs).collect()
}

fn fixture_set(v: &Value) -> BTreeSet<Labels> {
    v.as_array().unwrap().iter().map(simplex_list).collect()
}

/// The correct `all_connected_triangulations(only_fine=True)` counts for the
/// B1-affected entries, from `KNOWN_DIVERGENT.md`.
const B1_CORRECT: [(&str, usize); 6] = [
    ("octagon_vectors", 1),
    ("interior_point_triangle", 1),
    ("many_interior_points", 8),
    ("moderate_coords", 1),
    ("duplicate_points", 0),
    ("square_plus_axes", 64),
];

#[derive(Default, Debug)]
struct Tally {
    matched: usize,
    triangulations: usize,
    rejected: usize,
    b1: usize,
    skipped: usize,
}

fn corpus_entry<K: Kind>(entry: &Value, tally: &mut Tally) {
    let name = entry["name"].as_str().unwrap();
    let config: Config<K> = build(entry["matrix"].as_array().unwrap());
    let untrusted = untrusted(entry);
    let results = &entry["results"];
    for (method, only_fine) in [
        ("all_connected_triangulations", false),
        ("all_connected_triangulations_fine", true),
    ] {
        let expected = &results[method];
        let b1 = untrusted.contains(&method);
        let known = B1_CORRECT.iter().any(|(n, _)| *n == name);
        if (b1 && !known) || (!b1 && expected.get("timeout").is_some()) {
            // `vc_degenerate_flip`: the oracle gave up (and its fine result
            // is B1-untrusted with no known value), and the component is far
            // too large for a default test run.
            tally.skipped += 1;
            continue;
        }
        let ours = config
            .chirotope()
            .and_then(|chi| triang::all_connected(&chi, only_fine).map(|ts| (chi, ts)));
        if let Some(err) = expected.get("error").and_then(Value::as_str) {
            assert!(err.contains("not full rank"), "{name}: {err}");
            assert!(
                matches!(ours, Err(Error::NotFullRank { .. })),
                "{name}: {method}"
            );
            tally.rejected += 1;
            continue;
        }
        let (chi, ours) = ours.unwrap_or_else(|e| panic!("{name}: {method}: {e}"));
        let mut sorted = ours.clone();
        sorted.sort();
        assert_eq!(sorted, ours, "{name}: canonical order");
        if ours.len() <= CHECK_LIMIT {
            for t in &ours {
                assert_eq!(t.check(&chi), Ok(()), "{name}: {method}");
            }
        }
        if only_fine {
            assert!(ours.iter().all(Triangulation::is_fine), "{name}");
        }
        if b1 {
            // B1: assert the known-correct value instead.
            let (_, correct) = B1_CORRECT
                .iter()
                .find(|(n, _)| *n == name)
                .unwrap_or_else(|| panic!("{name}: untrusted without a known value"));
            assert_eq!(ours.len(), *correct, "{name}: B1");
            let all_fine = &results["all_triangulations_fine"]["ok"];
            assert_eq!(as_set(&ours), fixture_set(all_fine), "{name}: B1 set");
            tally.b1 += 1;
        } else {
            let ok = expected
                .get("ok")
                .unwrap_or_else(|| panic!("{name}: {method}: {expected}"));
            let theirs: Vec<Labels> = ok.as_array().unwrap().iter().map(simplex_list).collect();
            let ours: Vec<Labels> = ours.iter().map(Triangulation::to_vecs).collect();
            assert_eq!(ours, theirs, "{name}: {method}");
            tally.matched += 1;
        }
        tally.triangulations += ours.len();
    }
}

#[test]
fn corpus_and_b1() {
    let corpus = corpus();
    let mut tally = Tally::default();
    for entry in corpus["entries"].as_array().unwrap() {
        match entry["kind"].as_str().unwrap() {
            "point" => corpus_entry::<Points>(entry, &mut tally),
            "vector" => corpus_entry::<Vectors>(entry, &mut tally),
            k => panic!("unknown kind {k}"),
        }
    }
    println!("{tally:?}");
    assert_eq!(tally.b1, B1_CORRECT.len());
    // 31 entries, two methods each; both of vc_degenerate_flip's are skipped.
    assert_eq!(
        tally.matched + tally.b1 + tally.rejected + tally.skipped,
        62
    );
    assert_eq!(tally.skipped, 2);
}

/// `vc_degenerate_flip`'s `only_fine` result is marked untrusted (B1) but has
/// no known value; it is the one entry the corpus test leaves out entirely.
#[test]
fn every_b1_entry_has_a_known_value_but_one() {
    let corpus = corpus();
    let unknown: Vec<&str> = corpus["entries"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|e| untrusted(e).contains(&"all_connected_triangulations_fine"))
        .map(|e| e["name"].as_str().unwrap())
        .filter(|n| !B1_CORRECT.iter().any(|(m, _)| m == n))
        .collect();
    assert_eq!(unknown, vec!["vc_degenerate_flip"]);
}

#[derive(Default, Debug)]
struct TopcomTally {
    counts: usize,
    fine_counts: usize,
    no_fine_seed: usize,
    sets: usize,
    triangulations: usize,
}

fn topcom_entry<K: Kind>(entry: &Value, limit: u64, tally: &mut TopcomTally) {
    let name = entry["name"].as_str().unwrap();
    let config: Config<K> = build(entry["matrix"].as_array().unwrap());
    let chi: Chirotope = config.chirotope().unwrap();
    if let Some(n) = entry["n_connected"].as_u64().filter(|&n| n <= limit) {
        let ours = triang::all_connected(&chi, false).unwrap();
        assert_eq!(ours.len() as u64, n, "{name}: points2ntriangs");
        if let Some(set) = entry.get("connected") {
            assert_eq!(as_set(&ours), fixture_set(set), "{name}: points2triangs");
            for t in &ours {
                assert_eq!(t.check(&chi), Ok(()), "{name}");
            }
            tally.sets += 1;
        }
        tally.counts += 1;
        tally.triangulations += ours.len();
    }
    if let Some(n) = entry["n_connected_fine"].as_u64().filter(|&n| n <= limit) {
        let mut count = 0;
        for t in FlipGraph::new(&chi, true).unwrap() {
            assert!(t.unwrap().is_fine(), "{name}");
            count += 1;
        }
        if entry["fine_seed_not_fine"].as_bool().unwrap() {
            // TOPCOM walked from a non-fine seed; there are no fine
            // triangulations at all.
            assert_eq!(count, 0, "{name}: no fine triangulation exists");
            tally.no_fine_seed += 1;
        } else {
            assert_eq!(count as u64, n, "{name}: points2nfinetriangs");
            tally.fine_counts += 1;
        }
        tally.triangulations += count;
    }
}

fn topcom_counts(limit: u64) -> TopcomTally {
    let data = fixture("topcom.json.gz");
    let mut tally = TopcomTally::default();
    for entry in data["entries"].as_array().unwrap() {
        match entry["kind"].as_str().unwrap() {
            "point" => topcom_entry::<Points>(entry, limit, &mut tally),
            "vector" => topcom_entry::<Vectors>(entry, limit, &mut tally),
            k => panic!("unknown kind {k}"),
        }
    }
    println!("{tally:?}");
    tally
}

#[test]
fn small_components_match_topcom() {
    let tally = topcom_counts(DEFAULT_LIMIT);
    assert!(tally.counts >= 300, "{tally:?}");
    assert_eq!(tally.sets, 302);
}

#[test]
#[ignore = "about four minutes in release mode; run with --release -- --ignored"]
fn all_components_match_topcom() {
    let tally = topcom_counts(u64::MAX);
    assert_eq!(tally.counts, 442);
    assert_eq!(tally.fine_counts + tally.no_fine_seed, 451);
}
