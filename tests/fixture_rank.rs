//! `Config::rank` against the C++ oracle, over the whole fixture corpus.
//!
//! The corpus does not record ranks directly, but every result implies one:
//!
//! * a triangulation the oracle returned is pure, and each of its simplices
//!   has exactly `rank` labels, with `rank` equal to the number of rows of the
//!   homogenised matrix (TOPCOM only accepts full-rank input, and CGAL only
//!   produces well-formed output for it);
//! * `"Configuration is not full rank"` is the oracle's own exact-rank check
//!   failing.
//!
//! Results listed under `untrusted` are skipped, as `KNOWN_DIVERGENT.md`
//! requires. For the rank-deficient entries the affine-hull reduction is
//! checked too: it is what fixes bug B4.

mod common;

use common::{build, corpus, triangulations, untrusted};
use serde_json::Value;
use triangulumancer::config::{Config, Kind, PointConfig, VectorConfig};

#[derive(Default)]
struct Tally {
    entries: usize,
    simplices: usize,
    not_full_rank: usize,
    reduced: usize,
}

fn check<K: Kind>(entry: &Value, tally: &mut Tally) {
    let name = entry["name"].as_str().unwrap();
    let config: Config<K> = build(entry["matrix"].as_array().unwrap());
    let rank = config.rank();
    let rows = config.rows();
    let untrusted = untrusted(entry);

    for (method, result) in entry["results"].as_object().unwrap() {
        if untrusted.contains(&method.as_str()) {
            continue;
        }
        if let Some(ok) = result.get("ok") {
            for s in triangulations(method, ok).concat() {
                assert_eq!(
                    rank, rows,
                    "{name}: {method} triangulated a rank-deficient configuration"
                );
                assert_eq!(
                    s.len(),
                    rank,
                    "{name}: {method} simplex {s:?} vs rank {rank}"
                );
                tally.simplices += 1;
            }
        } else if let Some(err) = result.get("error").and_then(Value::as_str) {
            assert!(
                err.contains("not full rank"),
                "{name}: {method}: unexpected oracle error {err}"
            );
            assert!(
                rank < rows,
                "{name}: oracle says not full rank, rank {rank} of {rows}"
            );
            tally.not_full_rank += 1;
        }
        // Timeouts carry no information about the rank.
    }

    if rank < rows {
        let reduced = config.reduce_to_hull().unwrap();
        assert!(reduced.is_full_rank(), "{name}");
        assert_eq!(reduced.rank(), rank, "{name}");
        assert_eq!(reduced.len(), config.len(), "{name}");
        tally.reduced += 1;
    } else {
        assert_eq!(config.require_full_rank(), Ok(()), "{name}");
    }
    tally.entries += 1;
}

#[test]
fn rank_matches_the_oracle_on_the_whole_corpus() {
    let corpus = corpus();
    assert_eq!(corpus["oracle_version"], "0.3.0");
    let entries = corpus["entries"].as_array().unwrap();
    let mut tally = Tally::default();
    for entry in entries {
        match entry["kind"].as_str().unwrap() {
            "point" => check::<triangulumancer::config::Points>(entry, &mut tally),
            "vector" => check::<triangulumancer::config::Vectors>(entry, &mut tally),
            k => panic!("unknown kind {k}"),
        }
    }
    println!(
        "{} entries, {} simplices checked, {} not-full-rank results, {} reduced",
        tally.entries, tally.simplices, tally.not_full_rank, tally.reduced
    );
    assert_eq!(tally.entries, entries.len());
    assert!(tally.simplices > 1000);
    // rank_deficient_collinear, rank_deficient_coplanar_3d and single_point.
    assert_eq!(tally.reduced, 3);
}

/// `KNOWN_DIVERGENT.md`, B4: the correct answers live in the configuration's
/// own affine hull. The triangulations themselves are phase 6; the hulls are
/// checked here.
#[test]
fn rank_deficient_entries_reduce_to_their_affine_hull() {
    let collinear: PointConfig =
        build(&serde_json::from_str::<Vec<Value>>("[[0,0],[1,1],[2,2],[3,3]]").unwrap());
    let r = collinear.reduce_to_hull().unwrap();
    assert_eq!((r.dim(), r.to_row_major()), (1, vec![0, 1, 2, 3]));

    let single: PointConfig = build(&serde_json::from_str::<Vec<Value>>("[[0,0]]").unwrap());
    let r = single.reduce_to_hull().unwrap();
    assert_eq!((r.dim(), r.len()), (0, 1));

    let v: VectorConfig = build(&serde_json::from_str::<Vec<Value>>("[[1,2],[2,4]]").unwrap());
    assert_eq!(v.reduce_to_hull().unwrap().dim(), 1);
}
