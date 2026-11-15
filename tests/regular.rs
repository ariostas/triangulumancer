//! Regular and Delaunay triangulations (phase 6).
//!
//! * Against the C++ oracle over the fixture corpus: every trusted
//!   `delaunay_triangulation` and `triangulate_with_*` result must match
//!   exactly, in canonical order — including the cospherical and lattice
//!   cases, where the answer depends on the tie-breaking rule.
//! * The positive assertions of `tests/fixtures/KNOWN_DIVERGENT.md` for the
//!   results the oracle got wrong (B3, B4, B5).
//! * Random configurations, mostly degenerate, against the independent oracles
//!   of `tests/common/regular.rs`: the regularity certificate on every result,
//!   and on small inputs the brute-force regular triangulation of numerically
//!   perturbed lifts, which pins the tie-breaking too.
//!
//! Every result also passes the TDS invariant checks after every insertion,
//! which run in debug builds.

mod common;

use common::regular::{assert_regular, brute_force};
use common::{build, corpus, simplex_list, untrusted};
use dashu_int::IBig;
use serde_json::Value;
use triangulumancer::Error;
use triangulumancer::config::{PointConfig, Points};
use triangulumancer::regular::{self, Scalar, lift};
use triangulumancer::simplex::Simplex;
use triangulumancer::triang::Triangulation;

fn floats(v: &Value) -> Vec<Scalar> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| Scalar::Float(x.as_f64().unwrap()))
        .collect()
}

/// The method name of the corpus, the lifts, and our result.
fn run_method(config: &PointConfig, method: &str, params: &Value) -> (Vec<IBig>, Triangulation) {
    let lifts = if method == "delaunay_triangulation" {
        lift::delaunay(config)
    } else if let Some(p) = method.strip_prefix("triangulate_with_heights__") {
        lift::from_heights(config, &floats(&params[p])).unwrap()
    } else if let Some(p) = method.strip_prefix("triangulate_with_weights__") {
        lift::from_weights(config, &floats(&params[p])).unwrap()
    } else {
        unreachable!("{method}")
    };
    let t = regular::from_lifts(config, lifts.clone()).unwrap();
    (lifts, t)
}

fn regular_methods(entry: &Value) -> Vec<String> {
    entry["results"]
        .as_object()
        .unwrap()
        .keys()
        .filter(|k| k.starts_with("delaunay") || k.starts_with("triangulate_with"))
        .cloned()
        .collect()
}

#[test]
fn corpus_matches_the_oracle() {
    let corpus = corpus();
    let (mut matched, mut skipped) = (0, 0);
    for entry in corpus["entries"].as_array().unwrap() {
        if entry["kind"] != "point" {
            continue;
        }
        let name = entry["name"].as_str().unwrap();
        let config: PointConfig = build::<Points>(entry["matrix"].as_array().unwrap());
        let untrusted = untrusted(entry);
        for method in regular_methods(entry) {
            let (lifts, ours) = run_method(&config, &method, &entry["params"]);
            let what = format!("{name}: {method}");
            assert_regular(&config, &lifts, &ours, &what);
            if untrusted.contains(&method.as_str()) {
                skipped += 1;
                continue;
            }
            let expected = simplex_list(&entry["results"][&method]["ok"]);
            assert_eq!(ours.to_vecs(), expected, "{what}");
            matched += 1;
        }
    }
    println!("matched {matched}, untrusted {skipped}");
    assert!(matched >= 150, "{matched}");
}

/// Our result for a corpus entry and method, by name.
fn corpus_result(name: &str, method: &str) -> Triangulation {
    let corpus = corpus();
    let entry = corpus["entries"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["name"] == name)
        .unwrap();
    let config: PointConfig = build::<Points>(entry["matrix"].as_array().unwrap());
    run_method(&config, method, &entry["params"]).1
}

fn vecs(t: &Triangulation) -> Vec<Vec<usize>> {
    t.to_vecs()
}

/// `KNOWN_DIVERGENT.md`, B3: heights on coordinates past `46341`.
#[test]
fn known_divergent_b3() {
    for (name, method, expected) in [
        (
            "large_coords_b3",
            "triangulate_with_heights__alt",
            vec![vec![0, 1, 3], vec![0, 2, 3]],
        ),
        (
            "large_coords_b3",
            "triangulate_with_heights__ones",
            vec![vec![0, 1, 2], vec![1, 2, 3]],
        ),
        (
            "large_coords_b3",
            "triangulate_with_heights__ramp",
            vec![vec![0, 1, 2], vec![1, 2, 3]],
        ),
        (
            "moderate_coords",
            "triangulate_with_heights__alt",
            vec![vec![0, 1, 3], vec![0, 2, 3]],
        ),
        (
            "moderate_coords",
            "triangulate_with_heights__ones",
            vec![vec![0, 1, 4], vec![0, 2, 4], vec![1, 3, 4], vec![2, 3, 4]],
        ),
        (
            "moderate_coords",
            "triangulate_with_heights__ramp",
            vec![vec![0, 1, 2], vec![1, 2, 3]],
        ),
    ] {
        assert_eq!(
            vecs(&corpus_result(name, method)),
            expected,
            "{name}: {method}"
        );
    }

    // And a case where 32-bit squares really are wrong: weights on a square of
    // side 10^5, where |p|^2 = 10^10 overflows. With w = (0, 0, 0, 10^10 + 1),
    // the corner (1, 1) is lifted to -1, below the plane of the other three
    // (lifts 0, 10^10, 10^10 at (0, 0), (1, 0), (0, 1)), so the diagonal is
    // 0-3; with the overflowed square it was not.
    let s = 100_000;
    let square = PointConfig::from_row_major(&[0, 0, s, 0, 0, s, s, s], 4, 2).unwrap();
    let w = [0.into(), 0.into(), 0.into(), Scalar::from(2 * s * s + 1)];
    let t = regular::with_weights(&square, &w).unwrap();
    assert_eq!(vecs(&t), vec![vec![0, 1, 3], vec![0, 2, 3]]);
    // Near the top of the i64 range, exactly.
    let s = 1i64 << 62;
    let square = PointConfig::from_row_major(&[0, 0, s, 0, 0, s, s, s], 4, 2).unwrap();
    let t = regular::delaunay(&square).unwrap();
    assert_eq!(vecs(&t), vec![vec![0, 1, 2], vec![1, 2, 3]]);
    let t = regular::with_heights(&square, &[0.into(), 1.into(), 0.into(), 0.into()]).unwrap();
    assert_eq!(vecs(&t), vec![vec![0, 1, 3], vec![0, 2, 3]]);
}

/// `KNOWN_DIVERGENT.md`, B4: rank-deficient input.
#[test]
fn known_divergent_b4() {
    let line = [vec![0, 1], vec![1, 2], vec![2, 3]];
    for method in [
        "delaunay_triangulation",
        "triangulate_with_heights__ones",
        "triangulate_with_heights__ramp",
        "triangulate_with_weights__alt",
        "triangulate_with_weights__ones",
        "triangulate_with_weights__ramp",
    ] {
        let t = corpus_result("rank_deficient_collinear", method);
        assert_eq!(vecs(&t), line, "{method}");
    }
    // Heights (-1, 0, 1, -1): the middle points are above the segment 0-3.
    let t = corpus_result("rank_deficient_collinear", "triangulate_with_heights__alt");
    assert_eq!(vecs(&t), vec![vec![0, 3]]);
    assert_eq!(t.unused(), Simplex::from([1, 2]));

    for method in ["delaunay_triangulation", "triangulate_with_weights__ramp"] {
        let t = corpus_result("single_point", method);
        assert_eq!(vecs(&t), vec![vec![0]], "{method}");
    }

    // The coplanar unit square in 3D is a unit square in its own plane.
    let t = corpus_result("rank_deficient_coplanar_3d", "delaunay_triangulation");
    assert_eq!(vecs(&t), vec![vec![0, 1, 2], vec![1, 2, 3]]);
    assert_eq!(t.rank(), 3);
}

/// `KNOWN_DIVERGENT.md`, B5: duplicates resolve deterministically.
#[test]
fn known_divergent_b5() {
    // Points 1 and 3 coincide. Equal lifts: the lower index wins.
    for (method, survivor) in [
        ("delaunay_triangulation", 1),
        ("triangulate_with_heights__alt", 3),
        ("triangulate_with_heights__ones", 1),
        ("triangulate_with_heights__ramp", 1),
        ("triangulate_with_weights__alt", 1),
        ("triangulate_with_weights__ones", 1),
        ("triangulate_with_weights__ramp", 3),
    ] {
        let t = corpus_result("duplicate_points", method);
        let mut expected = vec![0, 2, survivor];
        expected.sort();
        assert_eq!(vecs(&t), vec![expected], "{method}");
        assert_eq!(t.unused(), Simplex::from([4 - survivor]), "{method}");
    }

    // Distinct points that collide as doubles, which 0.3 confused.
    let big = 1i64 << 60;
    let c = PointConfig::from_row_major(&[0, 0, big, 0, big + 1, 1, 0, big], 4, 2).unwrap();
    let t = regular::delaunay(&c).unwrap();
    assert!(t.is_fine(), "{:?}", vecs(&t));
    assert_regular(&c, &lift::delaunay(&c), &t, "near 2^60");
}

#[test]
fn errors_not_panics() {
    let c = PointConfig::from_row_major(&[0, 0, 1, 0, 0, 1], 3, 2).unwrap();
    assert!(matches!(
        regular::with_heights(&c, &[f64::NAN.into(), 0.into(), 0.into()]),
        Err(Error::InvalidParameter { index: 0, .. })
    ));
    assert!(matches!(
        regular::with_weights(&c, &[0.into()]),
        Err(Error::ParameterCountMismatch { .. })
    ));
    assert_eq!(
        regular::delaunay(&PointConfig::new()),
        Err(Error::EmptyConfiguration)
    );
}

// ---------------------------------------------------------------------------
// Random configurations
// ---------------------------------------------------------------------------

/// SplitMix64, as in `tests/sign_of_det.rs`.
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

/// A random point configuration: small coordinates (so lattice degeneracies
/// abound), sometimes embedded in a higher dimension (rank-deficient), and
/// sometimes with duplicates.
fn random_config(rng: &mut Rng, n: usize, d: usize, bound: i64) -> PointConfig {
    let ambient = d + rng.below(2) as usize;
    let mut rows: Vec<Vec<i64>> = Vec::with_capacity(n);
    for _ in 0..n {
        if !rows.is_empty() && rng.below(8) == 0 {
            let k = rng.below(rows.len() as u64) as usize;
            rows.push(rows[k].clone());
            continue;
        }
        let mut p: Vec<i64> = (0..d).map(|_| rng.signed(bound)).collect();
        // Embed: the extra coordinate is an affine function of the others.
        if ambient > d {
            p.push(p.iter().sum::<i64>() + 1);
        }
        rows.push(p);
    }
    PointConfig::from_row_major(&rows.concat(), n, ambient).unwrap()
}

fn random_lifts(rng: &mut Rng, config: &PointConfig) -> (&'static str, Vec<IBig>) {
    let n = config.len();
    match rng.below(4) {
        0 => ("delaunay", lift::delaunay(config)),
        1 => ("zero", vec![IBig::ZERO; n]),
        2 => {
            let h: Vec<Scalar> = (0..n).map(|_| Scalar::from(rng.signed(3))).collect();
            ("heights", lift::from_heights(config, &h).unwrap())
        }
        _ => {
            let w: Vec<Scalar> = (0..n)
                .map(|_| Scalar::Float(rng.signed(8) as f64 / 4.0))
                .collect();
            ("weights", lift::from_weights(config, &w).unwrap())
        }
    }
}

/// `cases` random configurations: certificate on all, brute force on those
/// small enough.
fn fuzz(seed: u64, cases: usize, max_n: usize, brute_n: usize) -> (usize, usize) {
    let mut rng = Rng(seed);
    let (mut certified, mut brute) = (0, 0);
    for case in 0..cases {
        let d = 1 + rng.below(3) as usize;
        let n = 1 + rng.below(max_n as u64) as usize;
        let bound = [1, 2, 3, 10][rng.below(4) as usize];
        let config = random_config(&mut rng, n, d, bound);
        let (kind, lifts) = random_lifts(&mut rng, &config);
        let what = format!("case {case} ({kind}), {config:?}, lifts {lifts:?}");
        let t =
            regular::from_lifts(&config, lifts.clone()).unwrap_or_else(|e| panic!("{what}: {e}"));
        assert_regular(&config, &lifts, &t, &what);
        certified += 1;
        if n <= brute_n {
            assert_eq!(t.to_vecs(), brute_force(&config, &lifts, 64), "{what}");
            brute += 1;
        }
    }
    (certified, brute)
}

#[test]
fn random_against_oracles() {
    let (certified, brute) = fuzz(0x5eed_0601, 300, 12, 9);
    println!("certified {certified}, brute-forced {brute}");
    assert!(brute > 100);
}

/// The long run: `cargo test --release --test regular -- --ignored`.
#[test]
#[ignore = "thousands of configurations; run in release"]
fn random_against_oracles_long() {
    let (certified, brute) = fuzz(0x5eed_0602, 20_000, 40, 10);
    println!("certified {certified}, brute-forced {brute}");
}

/// Larger random sets: certificate only.
#[test]
fn larger_random_sets() {
    let mut rng = Rng(0x5eed_0603);
    for d in 1..=4 {
        let n = [60, 50, 40, 25][d - 1];
        for bound in [3, 1000] {
            let config = random_config(&mut rng, n, d, bound);
            let (kind, lifts) = random_lifts(&mut rng, &config);
            let t = regular::from_lifts(&config, lifts.clone()).unwrap();
            assert_regular(&config, &lifts, &t, &format!("d = {d}, {kind}"));
        }
    }
}

/// Dimensions 5 to 7: the predicates' matrices are 5 x 5 and larger, so they
/// take the filter's `f64` fast path. Coordinates in `{-1, 0, 1}` or
/// `{-2, …, 2}` make nearly every test degenerate, so the fast path declines
/// and the exact fallback decides; `2^40` makes the lifts too wide for the
/// fast path altogether. Certificate only.
#[test]
fn high_dimensional_random_sets() {
    let mut rng = Rng(0x5eed_0604);
    let reps = if cfg!(debug_assertions) { 2 } else { 12 };
    for d in 5..=7 {
        for bound in [1, 2, 1 << 20, 1 << 40] {
            for _ in 0..reps {
                let n = d + 2 + rng.below(10) as usize;
                let config = random_config(&mut rng, n, d, bound);
                let (kind, lifts) = random_lifts(&mut rng, &config);
                let what = format!("d = {d}, {kind}, {config:?}, lifts {lifts:?}");
                let t = regular::from_lifts(&config, lifts.clone())
                    .unwrap_or_else(|e| panic!("{what}: {e}"));
                assert_regular(&config, &lifts, &t, &what);
            }
        }
    }
}

/// Peak resident set size of this process in MiB, from `/proc/self/status`;
/// `None` where that is unavailable.
fn peak_rss_mib() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|l| l.starts_with("VmHWM:"))?;
    let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kib / 1024)
}

/// A large Delaunay triangulation stays linear in memory: simplices with
/// labels of 128 or more must cost O(rank), not O(n), each. With an O(n)
/// bitset per simplex this run peaked at well over a gigabyte.
///
/// `cargo test --release --test regular -- --ignored large_delaunay`, alone,
/// since the peak is process-wide.
#[test]
#[ignore = "50 000 points; run in release, on its own"]
fn large_delaunay_memory_is_linear() {
    let mut rng = Rng(0x5eed_0604);
    let n = 50_000;
    let rows: Vec<i64> = (0..3 * n).map(|_| rng.signed(1_000_000)).collect();
    let config = PointConfig::from_row_major(&rows, n, 3).unwrap();
    let t = regular::delaunay(&config).unwrap();
    assert!(t.len() > 5 * n, "{} simplices", t.len());
    if let Some(peak) = peak_rss_mib() {
        println!("{} simplices, peak RSS {peak} MiB", t.len());
        assert!(peak < 400, "peak RSS {peak} MiB");
    }
}
