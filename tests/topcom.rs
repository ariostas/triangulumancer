//! The oriented-matroid layer and the placing/fine triangulations against
//! TOPCOM's own command-line tools.
//!
//! `tests/fixtures/topcom.json.gz` is produced by `scripts/gen_topcom_fixtures.py`
//! from `points2chiro`, `points2placingtriang` and `points2finetriang`, built
//! from the TOPCOM revision 0.3 used, over a few named and a few hundred
//! seeded random configurations, most of them degenerate.
//!
//! Besides matching TOPCOM, the triangulations are recomputed from a chirotope
//! *parsed from TOPCOM's text*, with no configuration anywhere: that is the
//! `PLAN.md` section 6.1 boundary — the algorithms see nothing but signs.
//!
//! Phase 3 adds circuits (`points2circuits`) and flip sets (`points2flips`,
//! fed the placing, fine and a few more triangulations as seeds), compared as
//! full sets of `(removed side, added side)` pairs.

mod common;

use std::collections::HashMap;

use common::{build, fixture, simplex_list};
use serde_json::Value;
use triangulumancer::arith::Sign;
use triangulumancer::config::{Config, Kind, Points, Vectors};
use triangulumancer::om::{self, OrientedMatroid, check_basis};
use triangulumancer::simplex::{Simplex, combinations};
use triangulumancer::triang::{self, Triangulation};

/// A chirotope given as a table, read from TOPCOM's `n,r:signs` format.
struct Table {
    n: usize,
    rank: usize,
    signs: HashMap<Simplex, Sign>,
}

impl Table {
    fn parse(text: &str) -> Self {
        let (header, body) = text.split_once(':').unwrap();
        let (n, rank) = header.split_once(',').unwrap();
        let (n, rank): (usize, usize) = (n.trim().parse().unwrap(), rank.trim().parse().unwrap());
        let signs: HashMap<Simplex, Sign> = combinations(n, rank)
            .zip(body.chars().filter(|c| !c.is_whitespace()))
            .map(|(b, c)| {
                let s = match c {
                    '+' => Sign::Positive,
                    '-' => Sign::Negative,
                    '0' => Sign::Zero,
                    _ => panic!("bad sign {c}"),
                };
                (b, s)
            })
            .collect();
        Self { n, rank, signs }
    }
}

impl OrientedMatroid for Table {
    fn n(&self) -> usize {
        self.n
    }

    fn rank(&self) -> usize {
        self.rank
    }

    fn sign(&self, basis: &Simplex) -> triangulumancer::Result<Sign> {
        check_basis(basis, self.n, self.rank)?;
        Ok(self.signs[basis])
    }
}

#[derive(Default, Debug)]
struct Tally {
    entries: usize,
    chirotopes: usize,
    bases: usize,
    simplices: usize,
    circuit_sets: usize,
    circuits: usize,
    flip_sets: usize,
    flips: usize,
}

/// A JSON list of `[positive, negative]` pairs of label lists.
fn pairs(v: &Value) -> Vec<(Vec<usize>, Vec<usize>)> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|p| {
            let side = |i: usize| -> Vec<usize> {
                p[i].as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_u64().unwrap() as usize)
                    .collect()
            };
            (side(0), side(1))
        })
        .collect()
}

fn circuits_and_flips(name: &str, chi: &impl OrientedMatroid, entry: &Value, tally: &mut Tally) {
    if let Some(expected) = entry.get("circuits") {
        let mut ours: Vec<_> = om::circuits(chi)
            .unwrap()
            .iter()
            .map(|z| (z.positive().to_vec(), z.negative().to_vec()))
            .collect();
        ours.sort();
        assert_eq!(ours, pairs(expected), "{name}: circuits");
        tally.circuit_sets += 1;
        tally.circuits += ours.len();
    }
    for record in entry["flips"].as_array().unwrap() {
        let labels: Vec<i64> = simplex_list(&record["triangulation"])
            .concat()
            .iter()
            .map(|&x| x as i64)
            .collect();
        let t = Triangulation::from_labels(chi.n(), chi.rank(), &labels, chi.rank()).unwrap();
        assert_eq!(t.check(chi), Ok(()), "{name}: seed");
        let reps = triang::flip::flip_reps(chi, &t, false).unwrap();
        let ours: Vec<_> = reps
            .iter()
            .map(|r| (r.removed_side().to_vec(), r.added_side().to_vec()))
            .collect();
        assert_eq!(ours, pairs(&record["flips"]), "{name}: flips of {t:?}");
        for rep in &reps {
            let next = rep.expand(&t).apply(&t).unwrap();
            assert_eq!(next.check(chi), Ok(()), "{name}: {rep:?} of {t:?}");
        }
        tally.flip_sets += 1;
        tally.flips += ours.len();
    }
}

fn check<K: Kind>(entry: &Value, tally: &mut Tally) {
    let name = entry["name"].as_str().unwrap();
    let config: Config<K> = build(entry["matrix"].as_array().unwrap());
    let chi = config.chirotope().unwrap();
    let placing = triang::placing(&chi).unwrap();
    let fine = triang::fine(&chi).unwrap();
    assert_eq!(
        placing.to_vecs(),
        simplex_list(&entry["placing"]),
        "{name}: placing"
    );
    assert_eq!(fine.to_vecs(), simplex_list(&entry["fine"]), "{name}: fine");
    assert_eq!(placing.check(&chi), Ok(()), "{name}: placing");
    assert_eq!(fine.check(&chi), Ok(()), "{name}: fine");
    tally.simplices += placing.len() + fine.len();
    circuits_and_flips(name, &chi, entry, tally);

    if let Some(expected) = entry["chirotope"].as_str() {
        let ours: String = om::to_topcom_string(&chi)
            .unwrap()
            .split_whitespace()
            .collect();
        assert_eq!(ours, expected, "{name}: chirotope");
        tally.chirotopes += 1;
        tally.bases += expected.split_once(':').unwrap().1.len();

        let table = Table::parse(expected);
        assert_eq!(triang::placing(&table).unwrap(), placing, "{name}");
        assert_eq!(triang::fine(&table).unwrap(), fine, "{name}");
    }
    tally.entries += 1;
}

#[test]
fn chirotope_placing_and_fine_match_topcom() {
    let data = fixture("topcom.json.gz");
    let mut tally = Tally::default();
    for entry in data["entries"].as_array().unwrap() {
        match entry["kind"].as_str().unwrap() {
            "point" => check::<Points>(entry, &mut tally),
            "vector" => check::<Vectors>(entry, &mut tally),
            k => panic!("unknown kind {k}"),
        }
    }
    println!("{tally:?}");
    assert_eq!(tally.entries, 504);
    assert_eq!(tally.chirotopes, 498);
    assert_eq!(tally.circuit_sets, 464);
    assert_eq!(tally.flip_sets, 1662);
}

/// Hand-checkable chirotopes, independent of both TOPCOM and `sign_of_det`.
#[test]
fn small_chirotopes_by_hand() {
    // The unit square, counter-clockwise: every triple is counter-clockwise.
    let square = Config::<Points>::from_row_major(&[0, 0, 1, 0, 1, 1, 0, 1], 4, 2).unwrap();
    assert_eq!(
        om::to_topcom_string(&square.chirotope().unwrap()).unwrap(),
        "4,3:\n++++\n"
    );
    // Four points on a line, with the middle two swapped. Homogenised as
    // columns `(x, 1)`, the chirotope at `{i < j}` is the sign of
    // `det [[x_i, x_j], [1, 1]] = x_i - x_j`.
    let line = Config::<Points>::from_row_major(&[0, 2, 1, 3], 4, 1).unwrap();
    assert_eq!(
        om::to_topcom_string(&line.chirotope().unwrap()).unwrap(),
        "4,2:\n---+--\n"
    );
    // Vectors e1, e2, -e1, e1 + e2: det(e1, e2) = 1, det(e1, -e1) = 0,
    // det(e1, e1+e2) = 1, det(e2, -e1) = 1, det(e2, e1+e2) = -1,
    // det(-e1, e1+e2) = -1.
    let v = Config::<Vectors>::from_row_major(&[1, 0, 0, 1, -1, 0, 1, 1], 4, 2).unwrap();
    assert_eq!(
        om::to_topcom_string(&v.chirotope().unwrap()).unwrap(),
        "4,2:\n+0++--\n"
    );
}
