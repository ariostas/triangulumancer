//! Placing and fine triangulations, and the triangulation check, against the
//! C++ oracle over the whole fixture corpus.
//!
//! * `placing_triangulation` and `fine_triangulation` must match the oracle
//!   exactly — the same simplices, in the canonical order the fixtures are
//!   stored in — or fail where it failed ("not full rank").
//! * Every whole triangulation the oracle produced by any method must pass
//!   `Triangulation::check`, and simple corruptions of each must fail it.
//!
//! Results listed under `untrusted` are skipped (`KNOWN_DIVERGENT.md`).

mod common;

use common::{build, corpus, triangulations, untrusted};
use serde_json::Value;
use triangulumancer::Error;
use triangulumancer::config::{Config, Kind, Points, Vectors};
use triangulumancer::om::{Chirotope, OrientedMatroid};
use triangulumancer::simplex::{Simplex, SimplexSet};
use triangulumancer::triang::{self, Triangulation};

type Algorithm = fn(&Chirotope) -> triangulumancer::Result<Triangulation>;

const ALGORITHMS: [(&str, Algorithm); 2] = [
    ("placing_triangulation", |c| triang::placing(c)),
    ("fine_triangulation", |c| triang::fine(c)),
];

#[derive(Default, Debug)]
struct Tally {
    matched: usize,
    rejected: usize,
    checked: usize,
    corrupted: usize,
}

fn differential<K: Kind>(entry: &Value, tally: &mut Tally) {
    let name = entry["name"].as_str().unwrap();
    let config: Config<K> = build(entry["matrix"].as_array().unwrap());
    let untrusted = untrusted(entry);
    for (method, algorithm) in ALGORITHMS {
        if untrusted.contains(&method) {
            continue;
        }
        let expected = &entry["results"][method];
        let ours = config.chirotope().and_then(|c| algorithm(&c));
        if let Some(ok) = expected.get("ok") {
            let ours = ours.unwrap_or_else(|e| panic!("{name}: {method}: {e}"));
            assert_eq!(ours.to_vecs(), common::simplex_list(ok), "{name}: {method}");
            assert!(ours.check(&config.chirotope().unwrap()).is_ok());
            tally.matched += 1;
        } else if let Some(err) = expected.get("error").and_then(Value::as_str) {
            assert!(err.contains("not full rank"), "{name}: {err}");
            assert!(
                matches!(ours, Err(Error::NotFullRank { .. })),
                "{name}: {method}: {ours:?}"
            );
            tally.rejected += 1;
        } else {
            panic!("{name}: {method}: the oracle did not finish: {expected}");
        }
    }
}

#[test]
fn placing_and_fine_match_the_oracle() {
    let corpus = corpus();
    let mut tally = Tally::default();
    for entry in corpus["entries"].as_array().unwrap() {
        match entry["kind"].as_str().unwrap() {
            "point" => differential::<Points>(entry, &mut tally),
            "vector" => differential::<Vectors>(entry, &mut tally),
            k => panic!("unknown kind {k}"),
        }
    }
    println!("{tally:?}");
    // 31 entries, two methods each; three entries are rank-deficient.
    assert_eq!(tally.matched + tally.rejected, 62);
    assert_eq!(tally.rejected, 6);
}

/// "This fails in TOPCOM 1.1.2": the regression the fork's `FineTriang` patch
/// fixed (`python/tests/test_triangulation.py`).
#[test]
fn degenerate_vector_configuration_has_46_fine_simplices() {
    let corpus = corpus();
    let entry = corpus["entries"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["name"] == "vc_degenerate_flip")
        .unwrap();
    let config: Config<Vectors> = build(entry["matrix"].as_array().unwrap());
    let chi = config.chirotope().unwrap();
    let t = triang::fine(&chi).unwrap();
    assert_eq!(t.len(), 46);
    assert!(t.is_fine());
    assert_eq!(t.check(&chi), Ok(()));
}

fn to_triangulation(chi: &Chirotope, simplices: &[Vec<usize>]) -> Triangulation {
    let set: SimplexSet = simplices
        .iter()
        .map(|s| s.iter().copied().collect::<Simplex>())
        .collect();
    Triangulation::new(chi.n(), chi.rank(), set).unwrap()
}

fn assert_invalid(chi: &Chirotope, t: &Triangulation, what: &str) {
    match t.check(chi) {
        Err(Error::InvalidTriangulation(_)) => {}
        other => panic!("{what}: {:?} accepted: {other:?}", t.to_vecs()),
    }
}

fn check_all<K: Kind>(entry: &Value, tally: &mut Tally) {
    let name = entry["name"].as_str().unwrap();
    let config: Config<K> = build(entry["matrix"].as_array().unwrap());
    let Ok(chi) = config.chirotope() else {
        return; // rank-deficient: nothing the oracle returned is trusted here
    };
    let untrusted = untrusted(entry);
    let mut all = Vec::new();
    for (method, result) in entry["results"].as_object().unwrap() {
        if untrusted.contains(&method.as_str()) || method == "flips_of_placing" {
            continue;
        }
        let Some(ok) = result.get("ok") else { continue };
        for simplices in triangulations(method, ok) {
            let t = to_triangulation(&chi, &simplices);
            assert_eq!(t.check(&chi), Ok(()), "{name}: {method}: {simplices:?}");
            tally.checked += 1;
            all.push(t);
        }
    }

    // Corruptions: drop a simplex, or add one from another triangulation.
    let universe: SimplexSet = all.iter().flat_map(|t| t.iter().cloned()).collect();
    all.sort();
    all.dedup();
    for t in &all {
        for s in t.iter() {
            let mut fewer = t.simplices().clone();
            fewer.remove(s);
            let fewer = Triangulation::new(t.n(), t.rank(), fewer).unwrap();
            assert_invalid(&chi, &fewer, name);
            tally.corrupted += 1;
        }
        for s in universe.iter().filter(|s| !t.simplices().contains(s)) {
            let mut more = t.simplices().clone();
            more.insert(s.clone());
            let more = Triangulation::new(t.n(), t.rank(), more).unwrap();
            assert_invalid(&chi, &more, name);
            tally.corrupted += 1;
        }
    }
}

#[test]
fn every_oracle_triangulation_passes_the_check_and_corruptions_fail() {
    let corpus = corpus();
    let mut tally = Tally::default();
    for entry in corpus["entries"].as_array().unwrap() {
        match entry["kind"].as_str().unwrap() {
            "point" => check_all::<Points>(entry, &mut tally),
            "vector" => check_all::<Vectors>(entry, &mut tally),
            k => panic!("unknown kind {k}"),
        }
    }
    println!("{tally:?}");
    assert!(tally.checked > 1000, "{tally:?}");
    assert!(tally.corrupted > 10_000, "{tally:?}");
}
