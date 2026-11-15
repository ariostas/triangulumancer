//! Metamorphic tests for regular and Delaunay triangulations, on small,
//! degenerate configurations:
//!
//! * Delaunay is invariant under `x ↦ kx + b` (`k > 0`), up to coordinates
//!   near `2^60`;
//! * relabelling the points (without duplicates) permutes the result;
//! * an affine image with coordinates near the `i64` limit, with wide lifts,
//!   agrees with the regularity certificate and with brute force;
//! * `f64` weights over the whole exponent range either triangulate (and pass
//!   the certificate) or are rejected with `InvalidParameter`, never anything
//!   else.
//!
//! The default run is a few hundred configurations per property; the
//! `#[ignore]`d ones run ten times as many with other seeds
//! (`cargo test --release --test regular_invariance -- --ignored`).

mod common;

use common::regular::{assert_regular, brute_force};
use dashu_int::IBig;
use triangulumancer::config::PointConfig;
use triangulumancer::regular::{self, Scalar, lift};

struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
    fn signed(&mut self, bound: i64) -> i64 {
        self.below(2 * bound as u64 + 1) as i64 - bound
    }
}

fn rows_cfg(rows: &[Vec<i64>], d: usize) -> PointConfig {
    PointConfig::from_row_major(&rows.concat(), rows.len(), d).unwrap()
}

fn small_rows(rng: &mut Rng, n: usize, d: usize, bound: i64, dups: bool) -> Vec<Vec<i64>> {
    let mut rows: Vec<Vec<i64>> = Vec::new();
    let n = n.min(((2 * bound + 1) as usize).pow(d as u32));
    while rows.len() < n {
        if dups && !rows.is_empty() && rng.below(6) == 0 {
            let k = rng.below(rows.len() as u64) as usize;
            rows.push(rows[k].clone());
            continue;
        }
        let p: Vec<i64> = (0..d).map(|_| rng.signed(bound)).collect();
        if !dups && rows.contains(&p) {
            continue;
        }
        rows.push(p);
    }
    rows
}

/// Delaunay is invariant under `x ↦ kx + b` (`k > 0`): the lifts change by
/// `k²` and an affine function, and the lexicographic order that breaks ties
/// is preserved.
fn delaunay_scale_translate_invariance(seed: u64, cases: usize) {
    let mut rng = Rng(seed);
    for case in 0..cases {
        let d = 1 + rng.below(3) as usize;
        let n = 1 + rng.below(12) as usize;
        let bound = [1, 2, 3][rng.below(3) as usize];
        let rows = small_rows(&mut rng, n, d, bound, true);
        let k: i64 = [1, 2, 7, 1 << 20, 1 << 40, 1 << 56][rng.below(6) as usize];
        let b: Vec<i64> = (0..d).map(|_| rng.signed(1 << 30) * (1 << 28)).collect();
        let big: Vec<Vec<i64>> = rows
            .iter()
            .map(|r| r.iter().zip(&b).map(|(&x, &bb)| x * k + bb).collect())
            .collect();
        let t1 = regular::delaunay(&rows_cfg(&rows, d)).unwrap();
        let t2 = regular::delaunay(&rows_cfg(&big, d))
            .unwrap_or_else(|e| panic!("case {case}: {e} for {big:?}"));
        assert_eq!(
            t1.to_vecs(),
            t2.to_vecs(),
            "case {case}: k = {k}, b = {b:?}, rows {rows:?}"
        );
    }
}

/// Relabelling points without duplicates must permute the result (insertion
/// order changes, perturbation by coordinates does not).
fn permutation_equivariance(seed: u64, cases: usize) {
    let mut rng = Rng(seed);
    for case in 0..cases {
        let d = 1 + rng.below(4) as usize;
        let n = 1 + rng.below(14) as usize;
        let bound = [1, 2, 3][rng.below(3) as usize];
        let rows = small_rows(&mut rng, n, d, bound, false);
        let lifts: Vec<IBig> = match rng.below(3) {
            0 => lift::delaunay(&rows_cfg(&rows, d)),
            1 => vec![IBig::ZERO; rows.len()],
            _ => (0..rows.len()).map(|_| IBig::from(rng.signed(2))).collect(),
        };
        let n = rows.len();
        let mut perm: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = rng.below(i as u64 + 1) as usize;
            perm.swap(i, j);
        }
        // New label `perm[i]` gets old point `i`.
        let mut prow = vec![vec![]; n];
        let mut plift = vec![IBig::ZERO; n];
        for i in 0..n {
            prow[perm[i]] = rows[i].clone();
            plift[perm[i]] = lifts[i].clone();
        }
        let t1 = regular::from_lifts(&rows_cfg(&rows, d), lifts.clone()).unwrap();
        let t2 = regular::from_lifts(&rows_cfg(&prow, d), plift).unwrap();
        let mut mapped: Vec<Vec<usize>> = t1
            .to_vecs()
            .into_iter()
            .map(|s| {
                let mut s: Vec<usize> = s.into_iter().map(|i| perm[i]).collect();
                s.sort();
                s
            })
            .collect();
        mapped.sort();
        assert_eq!(
            mapped,
            t2.to_vecs(),
            "case {case}: rows {rows:?}, lifts {lifts:?}, perm {perm:?}"
        );
    }
}

/// Huge coordinates (near the i64 limit) with degenerate structure, against
/// the exact certificate and brute force with wide perturbation.
fn huge_coordinates_vs_brute_force(seed: u64, cases: usize) {
    let mut rng = Rng(seed);
    for case in 0..cases {
        let d = 1 + rng.below(3) as usize;
        let n = 1 + rng.below(8) as usize;
        let rows = small_rows(&mut rng, n, d, 2, true);
        let n = rows.len();
        // Big affine image: x -> A x + b with A integer, big entries.
        let scale = 1i64 << (40 + rng.below(20));
        let a: Vec<i64> = (0..d * d).map(|_| rng.signed(3)).collect();
        let big: Vec<Vec<i64>> = rows
            .iter()
            .map(|r| {
                (0..d)
                    .map(|i| {
                        let mut s: i64 = 0;
                        for j in 0..d {
                            s += a[i * d + j] * r[j];
                        }
                        s * (scale / 16) + (i as i64 + 1) * (1i64 << 61)
                    })
                    .collect()
            })
            .collect();
        let c = rows_cfg(&big, d);
        let lifts: Vec<IBig> = match rng.below(3) {
            0 => lift::delaunay(&c),
            1 => (0..n).map(|_| IBig::from(rng.signed(2)) << 300).collect(),
            _ => (0..n).map(|_| IBig::from(rng.signed(2))).collect(),
        };
        let what = format!("case {case}: {big:?} lifts {lifts:?}");
        let t = match regular::from_lifts(&c, lifts.clone()) {
            Ok(t) => t,
            Err(e) => {
                // Overflow in hull reduction is allowed; nothing else.
                assert!(
                    matches!(e, triangulumancer::Error::Overflow(_)),
                    "{what}: {e}"
                );
                continue;
            }
        };
        assert_regular(&c, &lifts, &t, &what);
        assert_eq!(t.to_vecs(), brute_force(&c, &lifts, 600), "{what}");
    }
}

/// Float weights with extreme exponents: certificate only.
fn extreme_float_weights(seed: u64, cases: usize) {
    let mut rng = Rng(seed);
    for case in 0..cases {
        let d = 1 + rng.below(3) as usize;
        let n = 2 + rng.below(10) as usize;
        let rows = small_rows(&mut rng, n, d, 3, true);
        let c = rows_cfg(&rows, d);
        let w: Vec<Scalar> = (0..rows.len())
            .map(|_| {
                let e = rng.signed(300) as i32;
                Scalar::Float((rng.signed(1000) as f64) * 10f64.powi(e))
            })
            .collect();
        let what = format!("case {case}: {rows:?} {w:?}");
        match regular::with_weights(&c, &w) {
            Ok(t) => {
                let lifts = lift::from_weights(&c, &w).unwrap();
                assert_regular(&c, &lifts, &t, &what);
            }
            Err(e) => assert!(
                matches!(e, triangulumancer::Error::InvalidParameter { .. }),
                "{what}: {e}"
            ),
        }
    }
}

#[test]
fn delaunay_is_similarity_invariant() {
    delaunay_scale_translate_invariance(42, 300);
}

#[test]
fn relabelling_permutes() {
    permutation_equivariance(7, 400);
}

#[test]
fn huge_coordinates() {
    huge_coordinates_vs_brute_force(99, 300);
}

#[test]
fn float_weights_over_the_exponent_range() {
    extreme_float_weights(1234, 300);
}

#[test]
#[ignore = "thousands of configurations; run in release"]
fn invariance_long() {
    delaunay_scale_translate_invariance(43, 3000);
    permutation_equivariance(8, 4000);
    huge_coordinates_vs_brute_force(100, 3000);
    extreme_float_weights(1235, 3000);
}
