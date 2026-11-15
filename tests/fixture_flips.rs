//! Flips and neighbours of the placing triangulation against the C++ oracle
//! over the whole fixture corpus.
//!
//! The comparison is of full flip sets — every flip's removed and added
//! simplices — not counts, in the canonical order the fixtures are stored in.
//! Every neighbour must also pass `Triangulation::check`, and flipping back
//! across the inverse flip must return the placing triangulation.

mod common;

use common::{build, corpus, simplex_list, untrusted};
use serde_json::Value;
use triangulumancer::Error;
use triangulumancer::config::{Config, Kind, Points, Vectors};
use triangulumancer::triang;

/// A flip as its removed and added simplices, as label lists.
type FlipLists = (Vec<Vec<usize>>, Vec<Vec<usize>>);

#[derive(Default, Debug)]
struct Tally {
    entries: usize,
    flips: usize,
    neighbors: usize,
    rejected: usize,
}

fn differential<K: Kind>(entry: &Value, tally: &mut Tally) {
    let name = entry["name"].as_str().unwrap();
    let config: Config<K> = build(entry["matrix"].as_array().unwrap());
    let untrusted = untrusted(entry);
    for method in ["flips_of_placing", "neighbors_of_placing"] {
        assert!(!untrusted.contains(&method), "{name}: {method} untrusted");
        let expected = &entry["results"][method];
        let ours = config.chirotope().and_then(|chi| {
            let t = triang::placing(&chi)?;
            Ok((chi, t))
        });
        if let Some(err) = expected.get("error").and_then(Value::as_str) {
            assert!(err.contains("not full rank"), "{name}: {err}");
            assert!(
                matches!(ours, Err(Error::NotFullRank { .. })),
                "{name}: {method}"
            );
            tally.rejected += 1;
            continue;
        }
        let ok = expected
            .get("ok")
            .unwrap_or_else(|| panic!("{name}: {method}: the oracle did not finish"));
        let (chi, t) = ours.unwrap();
        if method == "flips_of_placing" {
            let ours: Vec<FlipLists> = triang::flips(&chi, &t)
                .unwrap()
                .iter()
                .map(|f| (f.removed().to_vecs(), f.added().to_vecs()))
                .collect();
            let theirs: Vec<FlipLists> = ok
                .as_array()
                .unwrap()
                .iter()
                .map(|f| (simplex_list(&f["pre"]), simplex_list(&f["post"])))
                .collect();
            assert_eq!(ours, theirs, "{name}: flips");
            tally.flips += ours.len();
        } else {
            let neighbors = triang::neighbors(&chi, &t).unwrap();
            let ours: Vec<Vec<Vec<usize>>> = neighbors.iter().map(|n| n.to_vecs()).collect();
            let theirs: Vec<Vec<Vec<usize>>> =
                ok.as_array().unwrap().iter().map(simplex_list).collect();
            assert_eq!(ours, theirs, "{name}: neighbors");
            for n in &neighbors {
                assert_eq!(n.check(&chi), Ok(()), "{name}: neighbor {n:?}");
                assert!(
                    triang::neighbors(&chi, n).unwrap().contains(&t),
                    "{name}: flipping back"
                );
            }
            tally.neighbors += neighbors.len();
        }
    }
    tally.entries += 1;
}

#[test]
fn flips_and_neighbors_of_placing_match_the_oracle() {
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
    assert_eq!(tally.entries, 31);
    assert_eq!(tally.rejected, 6);
    assert_eq!(tally.flips, tally.neighbors);
}
