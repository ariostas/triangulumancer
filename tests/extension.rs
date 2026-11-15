//! The extension-graph enumeration (`all_triangulations`) against the C++
//! oracle, TOPCOM's own enumerators, and the flip graph.
//!
//! * The core corpus, in canonical order, for every trusted result the oracle
//!   finished.
//! * `topcom.json.gz`: counts from `points2nalltriangs` and
//!   `points2nallfinetriangs`, and whole sets from `points2alltriangs` /
//!   `points2allfinetriangs` where they are small. The default run is bounded
//!   to results of at most `DEFAULT_LIMIT` triangulations; the `#[ignore]`d
//!   test runs everything TOPCOM finished
//!   (`cargo test --release --test extension -- --ignored all_counts`).
//! * The flip graph: on every configuration here the flip graph is connected
//!   (TOPCOM's own two enumerators agree on every one where both finished),
//!   so the two enumerations must give the same set, `only_fine` included.
//! * `vc_degenerate_flip`, which the oracle could not enumerate at all, is
//!   counted by an `#[ignore]`d test (`vc_degenerate_flip_finishes`).
//!
//! Every enumerated triangulation of the smaller results is also `check`ed.

mod common;

use std::collections::BTreeSet;

use common::{build, corpus, fixture, simplex_list, untrusted};
use serde_json::Value;
use triangulumancer::Error;
use triangulumancer::config::{Config, Kind, Points, VectorConfig, Vectors};
use triangulumancer::om::{Chirotope, OrientedMatroid};
use triangulumancer::triang::{self, ExtensionGraph, Triangulation};

/// Results larger than this are only enumerated by the ignored test.
const DEFAULT_LIMIT: u64 = 1000;

/// The flip graph is compared only on results up to this size, even in the
/// ignored test: it is about a hundred times slower per triangulation.
const CROSS_LIMIT: u64 = 20_000;

/// Triangulations are `check`ed only in results up to this size.
const CHECK_LIMIT: usize = 500;

type Labels = Vec<Vec<usize>>;

fn as_set(ts: &[Triangulation]) -> BTreeSet<Labels> {
    ts.iter().map(Triangulation::to_vecs).collect()
}

fn fixture_set(v: &Value) -> BTreeSet<Labels> {
    v.as_array().unwrap().iter().map(simplex_list).collect()
}

#[derive(Default, Debug)]
struct Tally {
    matched: usize,
    triangulations: usize,
    rejected: usize,
    skipped: usize,
}

fn corpus_entry<K: Kind>(entry: &Value, tally: &mut Tally) {
    let name = entry["name"].as_str().unwrap();
    let config: Config<K> = build(entry["matrix"].as_array().unwrap());
    let untrusted = untrusted(entry);
    let results = &entry["results"];
    for (method, only_fine) in [
        ("all_triangulations", false),
        ("all_triangulations_fine", true),
    ] {
        let expected = &results[method];
        assert!(!untrusted.contains(&method), "{name}: {method} untrusted");
        if expected.get("timeout").is_some() {
            // `vc_degenerate_flip`: see `vc_degenerate_flip_finishes`.
            tally.skipped += 1;
            continue;
        }
        let ours = config
            .chirotope()
            .and_then(|chi| triang::all_triangulations(&chi, only_fine).map(|ts| (chi, ts)));
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
        if ours.len() <= CHECK_LIMIT {
            for t in &ours {
                assert_eq!(t.check(&chi), Ok(()), "{name}: {method}");
            }
        }
        if only_fine {
            assert!(ours.iter().all(Triangulation::is_fine), "{name}");
        }
        let ok = expected
            .get("ok")
            .unwrap_or_else(|| panic!("{name}: {method}: {expected}"));
        let theirs: Vec<Labels> = ok.as_array().unwrap().iter().map(simplex_list).collect();
        let ours: Vec<Labels> = ours.iter().map(Triangulation::to_vecs).collect();
        assert_eq!(ours, theirs, "{name}: {method}");
        tally.matched += 1;
        tally.triangulations += ours.len();
    }
}

#[test]
fn matches_the_corpus() {
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
    // 31 entries, two methods each; both of vc_degenerate_flip's timed out.
    assert_eq!(tally.matched + tally.rejected + tally.skipped, 62);
    assert_eq!(tally.skipped, 2);
    assert_eq!(tally.rejected, 6);
}

/// The B1 values of `KNOWN_DIVERGENT.md` come from the oracle's
/// `all_triangulations(only_fine=True)`; the flip graph with B1 fixed must
/// give the same sets, which `tests/flipgraph.rs` asserts against the
/// fixture. Here, against this implementation directly.
#[test]
fn b1_values_agree_with_the_flip_graph() {
    let corpus = corpus();
    let mut checked = 0;
    for entry in corpus["entries"].as_array().unwrap() {
        if !untrusted(entry).contains(&"all_connected_triangulations_fine")
            || entry["name"] == "vc_degenerate_flip"
        {
            continue;
        }
        let chi = match entry["kind"].as_str().unwrap() {
            "point" => build::<Points>(entry["matrix"].as_array().unwrap()).chirotope(),
            _ => build::<Vectors>(entry["matrix"].as_array().unwrap()).chirotope(),
        }
        .unwrap();
        let ext = triang::all_triangulations(&chi, true).unwrap();
        let flip = triang::all_connected(&chi, true).unwrap();
        assert_eq!(ext, flip, "{}", entry["name"]);
        checked += 1;
    }
    assert_eq!(checked, 6);
}

#[derive(Default, Debug)]
struct TopcomTally {
    counts: usize,
    fine_counts: usize,
    sets: usize,
    triangulations: usize,
    connected: usize,
}

fn topcom_entry<K: Kind>(entry: &Value, limit: u64, tally: &mut TopcomTally) {
    let name = entry["name"].as_str().unwrap();
    let config: Config<K> = build(entry["matrix"].as_array().unwrap());
    let chi: Chirotope = config.chirotope().unwrap();
    for (only_fine, count, set, connected) in [
        (false, "n_all", "all", "n_connected"),
        (true, "n_all_fine", "all_fine", "n_connected_fine"),
    ] {
        let Some(n) = entry[count].as_u64().filter(|&n| n <= limit) else {
            continue;
        };
        let ours = triang::all_triangulations(&chi, only_fine).unwrap();
        assert_eq!(ours.len() as u64, n, "{name}: points2{count}");
        if only_fine {
            assert!(ours.iter().all(Triangulation::is_fine), "{name}");
            tally.fine_counts += 1;
        } else {
            tally.counts += 1;
        }
        if let Some(set) = entry.get(set) {
            assert_eq!(as_set(&ours), fixture_set(set), "{name}: {set}");
            tally.sets += 1;
        }
        if ours.len() <= CHECK_LIMIT {
            for t in &ours {
                assert_eq!(t.check(&chi), Ok(()), "{name}");
            }
        }
        tally.triangulations += ours.len();

        // The flip graph's component of the seed is in here, and since no
        // configuration in the corpus has a disconnected flip graph (TOPCOM's
        // own counts agree wherever both finished), it is all of it.
        if entry[connected]
            .as_u64()
            .is_some_and(|c| c <= limit.min(CROSS_LIMIT))
        {
            let flip = triang::all_connected(&chi, only_fine).unwrap();
            assert_eq!(flip, ours, "{name}: flip graph, only_fine = {only_fine}");
            tally.connected += 1;
        }
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
    println!("{tally:#?}");
    tally
}

// What the default run covers, so that a fixture change is noticed.
const COUNTS: usize = 388;
const FINE_COUNTS: usize = 432;
const SETS: usize = 716;
const CONNECTED: usize = 796;

#[test]
fn small_results_match_topcom() {
    let tally = topcom_counts(DEFAULT_LIMIT);
    assert_eq!(
        (tally.counts, tally.fine_counts, tally.sets, tally.connected),
        (COUNTS, FINE_COUNTS, SETS, CONNECTED),
        "{tally:?}"
    );
}

#[test]
#[ignore = "minutes in release mode; run with --release -- --ignored"]
fn all_counts_match_topcom() {
    let tally = topcom_counts(u64::MAX);
    // Every count TOPCOM finished: `n_all` is missing for the single
    // simplices and where `points2nalltriangs` timed out.
    let data = fixture("topcom.json.gz");
    let entries = data["entries"].as_array().unwrap();
    let have = |field: &str| entries.iter().filter(|e| e[field].is_u64()).count();
    assert_eq!(tally.counts, have("n_all"));
    assert_eq!(tally.fine_counts, have("n_all_fine"));
}

/// The 16 vectors of rank 4 from the 0.3 test suite (`test_flips.py`), on
/// which the oracle timed out in every enumeration and ran the machine out of
/// memory in `all_triangulations` (`KNOWN_DIVERGENT.md`).
fn vc_degenerate_flip() -> Chirotope {
    #[rustfmt::skip]
    let rows = [
        3, 1, -2, -1, 0, 0, 0, 1, -1, -3, 1, 1, 0, 2, 1, -1,
        2, 0, -1, 0, 1, -1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0,
        -1, 0, 0, 0, 1, 0, 0, 0, 0, -2, 1, 1, 3, 1, -1, -1,
        4, 3, -1, -2, 1, -1, 0, 1, 0, -3, 1, 2, 2, 2, 0, -1,
    ];
    VectorConfig::from_row_major(&rows, 16, 4)
        .unwrap()
        .chirotope()
        .unwrap()
}

#[test]
fn vc_degenerate_flip_tables() {
    let chi = vc_degenerate_flip();
    assert_eq!((chi.n(), chi.rank()), (16, 4));
    let g = ExtensionGraph::new(&chi, false).unwrap();
    // TOPCOM: "1434 valid simplices".
    assert_eq!(g.admissibles().len(), 1434);
    // The search streams: the first triangulations come at once.
    for t in g.take(1000) {
        assert!(!t.is_empty());
    }
}

/// The phase 5 headline target (`PLAN.md` section 5.4): every triangulation
/// of `vc_degenerate_flip`, counted in constant memory. Run it with
/// `cargo test --release --test extension -- --ignored vc_degenerate_flip`;
/// see `PLAN.md` section 13 for the time and memory it takes.
#[test]
#[ignore = "tens of minutes on several threads in release mode"]
fn vc_degenerate_flip_finishes() {
    let chi = vc_degenerate_flip();
    let threads = std::thread::available_parallelism().map_or(1, usize::from);
    let all = ExtensionGraph::new(&chi, false).unwrap().par_count(threads);
    let fine = ExtensionGraph::new(&chi, true).unwrap().par_count(threads);
    println!("vc_degenerate_flip: {all} triangulations, {fine} fine");
}
