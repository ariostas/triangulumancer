//! Differential test harness for `arith::sign_of_det`.
//!
//! Every determinant sign in the crate goes through `sign_of_det`, and a single
//! wrong sign silently changes the oriented matroid (PLAN.md section 7, phase 1).
//! This harness checks it, and each of its three tiers separately, against an
//! oracle that shares none of its code: a Laplace (cofactor) expansion over
//! arbitrary-precision integers. Plain cofactor expansion is `O(n!)`; the oracle
//! memoises the minors over column subsets, which is the same expansion in
//! `O(2^n n)` and keeps `n = 8` cheap enough for a million matrices.
//!
//! The oracle uses only multiplication and addition, never division, so it does
//! not share the exact-division step the Bareiss tiers depend on. Where entries
//! are small it is additionally recomputed in checked `i128` and the two oracle
//! evaluations are compared, so the bigint library is itself cross-checked.
//!
//! The default run is sized for debug builds. The full exit-criterion run of
//! PLAN.md is `#[ignore]`d:
//!
//! ```text
//! cargo test --release --test sign_of_det -- --ignored --nocapture
//! ```

use std::collections::BTreeMap;

use dashu_int::IBig;
use triangulumancer::arith::{self, Sign, SquareMatrix, bareiss, filter};

// ---------------------------------------------------------------------------
// Deterministic randomness, so that a failure is reproducible from its seed
// ---------------------------------------------------------------------------

/// SplitMix64. Self-contained so the harness needs no dev-dependency.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in `0..n`. The modulo bias is irrelevant here.
    fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }

    /// Uniform in `-bound..=bound`.
    fn signed(&mut self, bound: i64) -> i64 {
        let span = 2 * (bound as i128) + 1;
        ((self.next_u64() as u128 % span as u128) as i128 - bound as i128) as i64
    }

    fn any_i64(&mut self) -> i64 {
        self.next_u64() as i64
    }

    fn pick<T: Copy>(&mut self, xs: &[T]) -> T {
        xs[self.below(xs.len() as u64) as usize]
    }
}

// ---------------------------------------------------------------------------
// The oracle
// ---------------------------------------------------------------------------

/// The determinant of the row-major `n x n` matrix `m`, by Laplace expansion
/// along the last row of each leading block, memoised over column subsets.
///
/// `minor[S]`, for a set `S` of `k` columns, is the determinant of rows
/// `0..k` restricted to the columns in `S` (in increasing order). Expanding it
/// along its last row, row `k - 1`:
///
/// `minor[S] = sum_{j in S} (-1)^(k - 1 + pos_S(j)) * m[k-1][j] * minor[S \ {j}]`
fn oracle_det(m: &[i64], n: usize) -> IBig {
    let big: Vec<IBig> = m.iter().map(|&x| IBig::from(x)).collect();
    oracle_det_big(&big, n)
}

/// [`oracle_det`] over arbitrary-precision entries.
fn oracle_det_big(m: &[IBig], n: usize) -> IBig {
    let mut minor = vec![IBig::ZERO; 1 << n];
    minor[0] = IBig::ONE;
    for mask in 1usize..(1 << n) {
        let k = mask.count_ones() as usize;
        let row = &m[(k - 1) * n..k * n];
        let mut acc = IBig::ZERO;
        let mut pos = 0;
        for (j, entry) in row.iter().enumerate() {
            if mask & (1 << j) == 0 {
                continue;
            }
            if !entry.is_zero() {
                let term = entry * &minor[mask & !(1 << j)];
                if (k - 1 + pos) % 2 == 0 {
                    acc += term;
                } else {
                    acc -= term;
                }
            }
            pos += 1;
        }
        minor[mask] = acc;
    }
    minor[(1 << n) - 1].clone()
}

/// The same expansion in checked `i128`, or `None` on overflow.
fn oracle_det_i128(m: &[i64], n: usize) -> Option<i128> {
    let mut minor = vec![0i128; 1 << n];
    minor[0] = 1;
    for mask in 1usize..(1 << n) {
        let k = mask.count_ones() as usize;
        let row = &m[(k - 1) * n..k * n];
        let mut acc: i128 = 0;
        let mut pos = 0;
        for (j, &entry) in row.iter().enumerate() {
            if mask & (1 << j) == 0 {
                continue;
            }
            let term = (entry as i128).checked_mul(minor[mask & !(1 << j)])?;
            acc = if (k - 1 + pos) % 2 == 0 {
                acc.checked_add(term)?
            } else {
                acc.checked_sub(term)?
            };
            pos += 1;
        }
        minor[mask] = acc;
    }
    Some(minor[(1 << n) - 1])
}

fn sign_of(x: &IBig) -> Sign {
    if x.is_zero() {
        Sign::Zero
    } else if *x > IBig::ZERO {
        Sign::Positive
    } else {
        Sign::Negative
    }
}

// ---------------------------------------------------------------------------
// Matrix families
// ---------------------------------------------------------------------------

/// A test matrix, optionally with a determinant known by construction.
struct Case {
    m: Vec<i64>,
    n: usize,
    known_det: Option<IBig>,
}

impl Case {
    fn new(m: Vec<i64>, n: usize) -> Self {
        Self {
            m,
            n,
            known_det: None,
        }
    }
}

type Family = fn(&mut Rng, usize) -> Case;

/// Uniform entries with a magnitude drawn from a spread of scales.
fn uniform(rng: &mut Rng, n: usize) -> Case {
    let bound = rng.pick(&[1, 2, 3, 10, 100, 1 << 20, 1 << 31, 1 << 53, i64::MAX]);
    Case::new((0..n * n).map(|_| rng.signed(bound)).collect(), n)
}

/// Entries spread over the whole `i64` range: the i128 tier overflows almost
/// always from `n = 3` on, which forces the bigint fallback.
fn full_range(rng: &mut Rng, n: usize) -> Case {
    Case::new((0..n * n).map(|_| rng.any_i64()).collect(), n)
}

/// Entries drawn from the extremes of `i64`.
fn extremes(rng: &mut Rng, n: usize) -> Case {
    let vals = [i64::MIN, i64::MIN + 1, i64::MAX, i64::MAX - 1, -1, 0, 1];
    Case::new((0..n * n).map(|_| rng.pick(&vals)).collect(), n)
}

/// `B * C` with `B` of shape `n x k`, `C` of shape `k x n`, `k < n`: singular by
/// construction, with entries as large as `i64` allows for the chosen factors.
fn low_rank_matrix(rng: &mut Rng, n: usize, k: usize) -> Vec<i64> {
    // Keep each entry of the product within i64: k * bound^2 <= 2^62.
    let max_bits = (62 - (usize::BITS - k.max(1).leading_zeros()) as i64) / 2;
    let bits = rng.below(max_bits as u64 + 1) as i64;
    let bound = (1i64 << bits).max(1);
    let b: Vec<i64> = (0..n * k).map(|_| rng.signed(bound)).collect();
    let c: Vec<i64> = (0..k * n).map(|_| rng.signed(bound)).collect();
    let mut m = vec![0i64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc: i128 = 0;
            for t in 0..k {
                acc += b[i * k + t] as i128 * c[t * n + j] as i128;
            }
            m[i * n + j] = i64::try_from(acc).expect("bound keeps the product in i64");
        }
    }
    m
}

/// Rank-deficient: determinant exactly zero, entries up to `2^62`.
fn low_rank(rng: &mut Rng, n: usize) -> Case {
    if n == 0 {
        return Case::new(vec![], 0);
    }
    let k = rng.below(n as u64) as usize;
    let mut case = Case::new(low_rank_matrix(rng, n, k), n);
    case.known_det = Some(IBig::ZERO);
    case
}

/// A singular matrix of rank `n - 1` with one entry nudged by `±1`: the
/// determinant is a single cofactor, tiny next to the entries. This is the
/// case a floating-point filter is most likely to get wrong.
fn near_singular(rng: &mut Rng, n: usize) -> Case {
    if n == 0 {
        return Case::new(vec![], 0);
    }
    let mut m = low_rank_matrix(rng, n, n - 1);
    let idx = rng.below((n * n) as u64) as usize;
    let delta = if rng.below(2) == 0 { 1 } else { -1 };
    m[idx] = m[idx].saturating_add(delta);
    Case::new(m, n)
}

/// Entries just above `2^53`, where `i64 -> f64` conversion stops being exact,
/// on a near-singular pattern. Exercises the filter's handling of inexact
/// inputs.
fn around_2_53(rng: &mut Rng, n: usize) -> Case {
    let base = 1i64 << 53;
    let mut m: Vec<i64> = (0..n * n).map(|_| base + rng.signed(4)).collect();
    if n >= 2 {
        // Make two rows nearly equal.
        let (i, j) = (0, 1 + rng.below(n as u64 - 1) as usize);
        for c in 0..n {
            m[j * n + c] = m[i * n + c] + if c == 0 { rng.signed(1) } else { 0 };
        }
    }
    Case::new(m, n)
}

/// Mostly zeros, so every tier has to pivot.
fn sparse(rng: &mut Rng, n: usize) -> Case {
    let bound = rng.pick(&[1, 5, 1 << 40, i64::MAX]);
    let m = (0..n * n)
        .map(|_| {
            if rng.below(4) == 0 {
                rng.signed(bound)
            } else {
                0
            }
        })
        .collect();
    Case::new(m, n)
}

/// A diagonal matrix scrambled by random unimodular row and column operations,
/// stopping short of `i64` overflow. The determinant is known exactly by
/// construction, which checks the oracle as well as the implementation. Entries
/// grow large while the determinant stays tiny (often `0` or `±1`).
fn unimodular(rng: &mut Rng, n: usize) -> Case {
    let mut m = vec![0i64; n * n];
    let mut det = IBig::ONE;
    for i in 0..n {
        let d = rng.pick(&[0, 1, 1, 1, -1, 2, -3, 7]);
        m[i * n + i] = d;
        det *= IBig::from(d);
    }
    if n >= 2 {
        let target_bits = rng.below(63) as u32;
        let limit = 1i64.checked_shl(target_bits).unwrap_or(i64::MAX).max(2);
        for _ in 0..(4 * n * n) {
            let i = rng.below(n as u64) as usize;
            let j = (i + 1 + rng.below(n as u64 - 1) as usize) % n;
            let c = rng.signed(3);
            let by_row = rng.below(2) == 0;
            // row_i += c * row_j  (or the same on columns); det is unchanged.
            let mut next = m.clone();
            let mut ok = true;
            for t in 0..n {
                let (dst, src) = if by_row {
                    (i * n + t, j * n + t)
                } else {
                    (t * n + i, t * n + j)
                };
                match c
                    .checked_mul(m[src])
                    .and_then(|x| x.checked_add(m[dst]))
                    .filter(|x| x.unsigned_abs() <= limit as u64)
                {
                    Some(v) => next[dst] = v,
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            if ok {
                m = next;
            }
            if rng.below(8) == 0 {
                // A row swap negates the determinant.
                let k = (i + 1) % n;
                for t in 0..n {
                    m.swap(i * n + t, k * n + t);
                }
                det = -det;
            }
        }
    }
    Case {
        m,
        n,
        known_det: Some(det),
    }
}

/// Rows `(x, |x|^2, 1)` of integer points: the power-test matrix of PLAN.md
/// section 7, phase 6, which is the shape regular triangulations will feed in.
fn lifted(rng: &mut Rng, n: usize) -> Case {
    if n < 3 {
        return uniform(rng, n);
    }
    let bound = rng.pick(&[2, 10, 1000, 1 << 20, 1 << 30]);
    let d = n - 2;
    let mut m = Vec::with_capacity(n * n);
    for _ in 0..n {
        let x: Vec<i64> = (0..d).map(|_| rng.signed(bound)).collect();
        let norm: i128 = x.iter().map(|&v| v as i128 * v as i128).sum();
        m.extend_from_slice(&x);
        m.push(i64::try_from(norm).unwrap_or(i64::MAX));
        m.push(1);
    }
    Case::new(m, n)
}

/// Lattice points on a common sphere, lifted: every such matrix is singular,
/// because the lifted column is a multiple of the homogenising one. The
/// classic exact-zero case of Delaunay triangulations.
fn cospherical(rng: &mut Rng, n: usize) -> Case {
    if n < 3 {
        return low_rank(rng, n);
    }
    let d = n - 2;
    // Signed permutations of one vector all have the same norm.
    let base: Vec<i64> = (0..d).map(|_| rng.signed(1000)).collect();
    let norm: i64 = base.iter().map(|v| v * v).sum();
    let mut m = Vec::with_capacity(n * n);
    for _ in 0..n {
        let mut x = base.clone();
        for t in (1..d).rev() {
            let s = rng.below(t as u64 + 1) as usize;
            x.swap(t, s);
        }
        for v in &mut x {
            if rng.below(2) == 0 {
                *v = -*v;
            }
        }
        m.extend_from_slice(&x);
        m.push(norm);
        m.push(1);
    }
    let mut case = Case::new(m, n);
    case.known_det = Some(IBig::ZERO);
    case
}

/// A random matrix with two rows or two columns made equal, or one zeroed.
fn degenerate(rng: &mut Rng, n: usize) -> Case {
    let mut case = uniform(rng, n);
    if n >= 2 {
        let i = rng.below(n as u64) as usize;
        let j = (i + 1 + rng.below(n as u64 - 1) as usize) % n;
        let m = &mut case.m;
        match rng.below(4) {
            0 => (0..n).for_each(|t| m[j * n + t] = m[i * n + t]),
            1 => (0..n).for_each(|t| m[t * n + j] = m[t * n + i]),
            2 => (0..n).for_each(|t| m[i * n + t] = 0),
            _ => (0..n).for_each(|t| m[t * n + i] = 0),
        }
        case.known_det = Some(IBig::ZERO);
    }
    case
}

const FAMILIES: &[(&str, Family)] = &[
    ("uniform", uniform),
    ("full_range", full_range),
    ("extremes", extremes),
    ("low_rank", low_rank),
    ("near_singular", near_singular),
    ("around_2_53", around_2_53),
    ("sparse", sparse),
    ("unimodular", unimodular),
    ("lifted", lifted),
    ("cospherical", cospherical),
    ("degenerate", degenerate),
];

// ---------------------------------------------------------------------------
// The check
// ---------------------------------------------------------------------------

#[derive(Default, Debug)]
struct Stats {
    matrices: u64,
    /// Matrices the f64 filter decided.
    filter_hits: u64,
    /// Matrices the i128 tier decided.
    i128_hits: u64,
    /// Matrices where the i128 tier overflowed, i.e. the bigint tier was needed.
    i128_overflows: u64,
    /// Matrices with an exactly zero determinant.
    zeros: u64,
    /// Matrices whose oracle value was also checked in i128.
    oracle_cross_checks: u64,
}

fn transpose(m: &[i64], n: usize) -> Vec<i64> {
    let mut t = vec![0; n * n];
    for i in 0..n {
        for j in 0..n {
            t[j * n + i] = m[i * n + j];
        }
    }
    t
}

fn check(case: &Case, label: &str, stats: &mut Stats) {
    let (m, n) = (&case.m[..], case.n);
    let det = oracle_det(m, n);
    let ctx = || format!("{label}, n = {n}, matrix = {m:?}");

    if let Some(known) = &case.known_det {
        assert_eq!(&det, known, "oracle disagrees with construction: {}", ctx());
    }
    if let Some(small) = oracle_det_i128(m, n) {
        assert_eq!(det, IBig::from(small), "oracle i128 mismatch: {}", ctx());
        stats.oracle_cross_checks += 1;
    }
    let expected = sign_of(&det);

    let square = SquareMatrix::new(m, n).expect("well-formed matrix");
    assert_eq!(arith::sign_of_det(m, n).unwrap(), expected, "{}", ctx());
    assert_eq!(
        bareiss::sign_of_det_bigint(square),
        expected,
        "bigint tier: {}",
        ctx()
    );
    match bareiss::sign_of_det_i128(square) {
        Some(s) => {
            assert_eq!(s, expected, "i128 tier: {}", ctx());
            stats.i128_hits += 1;
        }
        None => stats.i128_overflows += 1,
    }
    if let Some(s) = filter::sign_of_det(square) {
        assert_eq!(s, expected, "f64 filter: {}", ctx());
        stats.filter_hits += 1;
    }

    // The entry-generic stack (phase 6) over the same values, as i64, i128
    // and IBig, must agree too — tier by tier.
    let wide: Vec<i128> = m.iter().map(|&x| x as i128).collect();
    let big: Vec<IBig> = m.iter().map(|&x| IBig::from(x)).collect();
    assert_eq!(arith::sign_of_det_of(m, n).unwrap(), expected, "{}", ctx());
    assert_eq!(
        arith::sign_of_det_of(&wide, n).unwrap(),
        expected,
        "{}",
        ctx()
    );
    assert_eq!(
        arith::sign_of_det_of(&big, n).unwrap(),
        expected,
        "{}",
        ctx()
    );
    if n >= 2 {
        assert_eq!(
            bareiss::sign_of_det_bigint_of(&big, n),
            expected,
            "{}",
            ctx()
        );
        if let Some(s) = bareiss::sign_of_det_i128_of(&wide, n) {
            assert_eq!(s, expected, "generic i128 tier: {}", ctx());
        }
        if let Some(s) = filter::sign_of_det_of(&big, n) {
            assert_eq!(s, expected, "generic f64 filter: {}", ctx());
        }
    }

    // Invariances: transposition preserves the sign, a row swap negates it.
    let t = transpose(m, n);
    assert_eq!(
        arith::sign_of_det(&t, n).unwrap(),
        expected,
        "transpose: {}",
        ctx()
    );
    if n >= 2 {
        let mut s = m.to_vec();
        for c in 0..n {
            s.swap(c, n + c);
        }
        assert_eq!(
            arith::sign_of_det(&s, n).unwrap(),
            -expected,
            "row swap: {}",
            ctx()
        );
    }

    stats.matrices += 1;
    if expected == Sign::Zero {
        stats.zeros += 1;
    }
}

/// Run `per_cell` matrices for every family and every size `0..=8`.
fn run(seed: u64, per_cell: usize) -> BTreeMap<&'static str, Stats> {
    let mut rng = Rng(seed);
    let mut all = BTreeMap::new();
    for &(name, family) in FAMILIES {
        let stats: &mut Stats = all.entry(name).or_default();
        for n in 0..=8 {
            for _ in 0..per_cell {
                let case = family(&mut rng, n);
                check(&case, name, stats);
            }
        }
    }
    all
}

fn report(all: &BTreeMap<&'static str, Stats>) -> Stats {
    let mut total = Stats::default();
    for (name, s) in all {
        println!(
            "{name:>14}: {:>8} matrices, {:>7} zero, f64 filter decided {:>8}, \
             i128 decided {:>8}, bigint needed {:>7}",
            s.matrices, s.zeros, s.filter_hits, s.i128_hits, s.i128_overflows
        );
        total.matrices += s.matrices;
        total.zeros += s.zeros;
        total.filter_hits += s.filter_hits;
        total.i128_hits += s.i128_hits;
        total.i128_overflows += s.i128_overflows;
        total.oracle_cross_checks += s.oracle_cross_checks;
    }
    println!("{:>14}: {total:?}", "total");
    total
}

fn assert_coverage(all: &BTreeMap<&'static str, Stats>, total: &Stats) {
    // Every tier must actually have been exercised, or the run proves nothing
    // about it.
    assert!(total.filter_hits > 0);
    assert!(total.i128_hits > 0);
    assert!(total.i128_overflows > 0);
    assert!(total.zeros > 0);
    assert!(total.oracle_cross_checks > 0);
    // The large-entry families exist to force the bigint fallback.
    assert!(all["full_range"].i128_overflows > 0);
    assert!(all["extremes"].i128_overflows > 0);
}

#[test]
fn oracle_sanity() {
    assert_eq!(oracle_det(&[], 0), IBig::ONE);
    assert_eq!(oracle_det(&[-4], 1), IBig::from(-4));
    assert_eq!(oracle_det(&[1, 2, 3, 4], 2), IBig::from(-2));
    // A 3-cycle permutation matrix is even; a transposition is odd.
    assert_eq!(oracle_det(&[0, 1, 0, 0, 0, 1, 1, 0, 0], 3), IBig::ONE);
    assert_eq!(oracle_det(&[0, 1, 0, 1, 0, 0, 0, 0, 1], 3), IBig::from(-1));
    assert_eq!(
        oracle_det(&[2, -1, 0, -1, 2, -1, 0, -1, 2], 3),
        IBig::from(4)
    );
}

#[test]
fn malformed_input_is_an_error_not_a_panic() {
    assert!(arith::sign_of_det(&[1, 2, 3], 2).is_err());
    assert!(SquareMatrix::new(&[1, 2, 3], 2).is_err());
    assert!(arith::sign_of_det(&[], 0).is_ok());
}

/// Entries wider than `i64`, as the regular-triangulation predicates produce
/// them: coordinate differences of up to 65 bits, and a last column of lifts
/// scaled by a power of two of up to a thousand bits. A third of the cases
/// are made singular by repeating a combination of rows, so the exact tiers
/// are exercised as well as the filter.
#[test]
fn wide_entries() {
    let mut rng = Rng(0x5eed_0006);
    let mut decided = [0usize; 3];
    for round in 0..3000 {
        let n = 2 + rng.below(6) as usize;
        let shift = rng.pick(&[0usize, 10, 60, 64, 120, 200, 1000]);
        let bound = rng.pick(&[3, 1000, 1 << 40, i64::MAX]);
        let mut m: Vec<IBig> = Vec::with_capacity(n * n);
        for _ in 0..n {
            for j in 0..n {
                let a = IBig::from(rng.signed(bound));
                if j + 1 == n {
                    let b = IBig::from(rng.signed(bound));
                    m.push((a * b) << shift);
                } else {
                    m.push(a - IBig::from(rng.signed(bound)));
                }
            }
        }
        if round % 3 == 0 {
            // Row n-1 := row 0 - row 1: exactly singular.
            for j in 0..n {
                m[(n - 1) * n + j] = &m[j] - &m[n + j];
            }
        }
        let expected = sign_of(&oracle_det_big(&m, n));
        let got = arith::sign_of_det_of(&m, n).unwrap();
        assert_eq!(got, expected, "n = {n}, shift = {shift}, {m:?}");
        assert_eq!(bareiss::sign_of_det_bigint_of(&m, n), expected);
        if let Some(s) = filter::sign_of_det_of(&m, n) {
            assert_eq!(s, expected);
            decided[0] += 1;
        }
        if let Some(wide) = m
            .iter()
            .map(|x| i128::try_from(x).ok())
            .collect::<Option<Vec<i128>>>()
        {
            assert_eq!(arith::sign_of_det_of(&wide, n).unwrap(), expected);
            if let Some(s) = bareiss::sign_of_det_i128_of(&wide, n) {
                assert_eq!(s, expected);
                decided[1] += 1;
            }
        }
        if expected == Sign::Zero {
            decided[2] += 1;
        }
    }
    // Every tier and the zero case were reached.
    assert!(decided.iter().all(|&k| k > 0), "{decided:?}");
    assert!(arith::sign_of_det_of(&[IBig::ONE; 3], 2).is_err());
}

/// The single tiers have no error to return for a matrix of the wrong size;
/// they panic, as documented, rather than read past it or answer for part of it.
#[test]
#[should_panic(expected = "do not make a 2 x 2 matrix")]
fn filter_tier_rejects_the_wrong_length() {
    let _ = filter::sign_of_det_of(&[1i64; 3], 2);
}

#[test]
#[should_panic(expected = "do not make a 2 x 2 matrix")]
fn i128_tier_rejects_the_wrong_length() {
    let _ = bareiss::sign_of_det_i128_of(&[1i64; 5], 2);
}

#[test]
#[should_panic(expected = "do not make a 2 x 2 matrix")]
fn bigint_tier_rejects_the_wrong_length() {
    let _ = bareiss::sign_of_det_bigint_of(&[1i64; 3], 2);
}

/// Adversarial for the filter's error bound: a singular matrix of rank
/// `n - 1`, up to `12 x 12`, with one entry moved by `±2^t` for every `t` from
/// `0` to `62`, so that the determinant (`±2^t` times a cofactor) sweeps from
/// far below the filter's error bound to far above it. The whole matrix is
/// also scaled by `2^shift` before the move, which makes the entries inexact
/// in `f64` (`shift >= 30` takes them past `2^53`) and, for the larger shifts,
/// wider than `i128`. The filter must decline or be right, on every entry
/// type; across the sweep it must do both for every size, or the sweep missed
/// its threshold.
#[test]
fn filter_threshold_sweep() {
    let mut rng = Rng(0x5eed_0007);
    let step = if cfg!(debug_assertions) { 4 } else { 1 };
    let max_n = if cfg!(debug_assertions) { 9 } else { 12 };
    let reps = if cfg!(debug_assertions) { 1 } else { 2 };
    let mut total = [0usize; 2];
    for n in 2..=max_n {
        let mut counts = [0usize; 2];
        for shift in [0u32, 30, 64, 100, 300] {
            for _ in 0..reps {
                let base = low_rank_matrix(&mut rng, n, n - 1);
                let idx = rng.below((n * n) as u64) as usize;
                for t in (0..63).step_by(step) {
                    for delta in [1i64 << t, -(1i64 << t)] {
                        let m: Vec<IBig> = base
                            .iter()
                            .enumerate()
                            .map(|(k, &x)| {
                                let v = IBig::from(x) << shift as usize;
                                if k == idx { v + delta } else { v }
                            })
                            .collect();
                        let expected = sign_of(&oracle_det_big(&m, n));
                        let ctx = || format!("n = {n}, shift = {shift}, t = {t}, {m:?}");
                        assert_eq!(arith::sign_of_det_of(&m, n).unwrap(), expected, "{}", ctx());
                        let mut answers = vec![filter::sign_of_det_of(&m, n)];
                        if let Some(wide) = m
                            .iter()
                            .map(|x| i128::try_from(x).ok())
                            .collect::<Option<Vec<i128>>>()
                        {
                            answers.push(filter::sign_of_det_of(&wide, n));
                            assert_eq!(arith::sign_of_det_of(&wide, n).unwrap(), expected);
                        }
                        if let Some(small) = m
                            .iter()
                            .map(|x| i64::try_from(x).ok())
                            .collect::<Option<Vec<i64>>>()
                        {
                            let square = SquareMatrix::new(&small, n).unwrap();
                            answers.push(filter::sign_of_det(square));
                            assert_eq!(arith::sign_of_det(&small, n).unwrap(), expected);
                        }
                        // The entry types convert to the same midpoints and
                        // radii, so they must also agree with each other.
                        assert!(answers.windows(2).all(|w| w[0] == w[1]), "{}", ctx());
                        match answers[0] {
                            Some(s) => {
                                assert_eq!(s, expected, "f64 filter: {}", ctx());
                                counts[0] += 1;
                            }
                            None => counts[1] += 1,
                        }
                    }
                }
            }
        }
        assert!(counts.iter().all(|&c| c > 0), "n = {n}: {counts:?}");
        total[0] += counts[0];
        total[1] += counts[1];
    }
    println!("filter decided {} and declined {}", total[0], total[1]);
}

/// The default run: a few hundred matrices per family and size.
#[test]
fn differential_default() {
    let per_cell = if cfg!(debug_assertions) { 200 } else { 2000 };
    let all = run(0x5eed_0001, per_cell);
    let total = report(&all);
    assert_coverage(&all, &total);
}

/// The phase 1 exit criterion: at least a million matrices.
#[test]
#[ignore = "about a million matrices; run with --release -- --ignored"]
fn differential_million() {
    let per_cell = 1_000_000usize.div_ceil(FAMILIES.len() * 9);
    let all = run(0x5eed_0002, per_cell);
    let total = report(&all);
    assert!(total.matrices >= 1_000_000);
    assert_coverage(&all, &total);
}
