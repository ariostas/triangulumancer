//! Both enumerations against a brute-force oracle that shares no code with the
//! crate, on small random point configurations full of degeneracies (lattice
//! points with coordinates in `{-2, ..., 2}`, and repeated points).
//!
//! The oracle lists every set of full-dimensional simplices whose volumes sum
//! to the volume of the hull and which pass an independent pseudo-manifold
//! test, using its own `i128` determinants. Against it:
//!
//! * `all_triangulations` must find exactly the oracle's triangulations, and
//!   with `only_fine` exactly its fine ones;
//! * `all_connected_triangulations` must too (the flip graph of every
//!   configuration this small is connected), every result must pass the
//!   oracle, and neighbourhood must be symmetric;
//! * `all_connected_triangulations(only_fine)` must be a subset of the fine
//!   ones, and in the plane, where the fine flip graph is connected, all of
//!   them;
//! * `Triangulation::check` must accept every one of the oracle's.
//!
//! The default run is 25 configurations; the `#[ignore]`d one runs
//! 600 (`cargo test --release --test brute_force -- --ignored`).

use std::collections::{BTreeSet, HashMap};

use triangulumancer::config::PointConfig;
use triangulumancer::simplex::{Simplex, SimplexSet, combinations};
use triangulumancer::triang::{self, Triangulation};

type Labels = Vec<Vec<usize>>;

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

/// A determinant by cofactor expansion; the matrices here are at most 4×4.
fn det(m: &[i128], n: usize) -> i128 {
    if n == 1 {
        return m[0];
    }
    let mut total = 0;
    for c in 0..n {
        let sub: Vec<i128> = (1..n)
            .flat_map(|r| (0..n).filter(move |&cc| cc != c).map(move |cc| (r, cc)))
            .map(|(r, cc)| m[r * n + cc])
            .collect();
        let term = m[c] * det(&sub, n - 1);
        total += if c % 2 == 0 { term } else { -term };
    }
    total
}

/// Homogenised points.
struct Points {
    rows: Vec<Vec<i128>>,
    rank: usize,
}

impl Points {
    fn det_of(&self, labels: &[usize]) -> i128 {
        let m: Vec<i128> = labels.iter().flat_map(|&i| self.rows[i].clone()).collect();
        det(&m, self.rank)
    }

    /// Whether `simplices` triangulate a region of volume `total`: no
    /// degenerate simplex, the volumes sum to `total`, and every facet either
    /// lies in two simplices on opposite sides of it or in one with no point
    /// beyond it.
    fn is_triangulation(&self, simplices: &[Vec<usize>], total: i128) -> bool {
        let vol: i128 = simplices.iter().map(|s| self.det_of(s).abs()).sum();
        if vol != total || simplices.iter().any(|s| self.det_of(s) == 0) {
            return false;
        }
        let mut facets: HashMap<Vec<usize>, Vec<usize>> = HashMap::new();
        for s in simplices {
            for &a in s {
                let f: Vec<usize> = s.iter().copied().filter(|&x| x != a).collect();
                facets.entry(f).or_default().push(a);
            }
        }
        facets.iter().all(|(f, apexes)| {
            let side = |x: usize| {
                let mut l = f.clone();
                l.push(x);
                self.det_of(&l).signum()
            };
            match apexes[..] {
                [a] => {
                    let s = side(a);
                    (0..self.rows.len()).all(|x| side(x) != -s)
                }
                [a, b] => side(a) == -side(b),
                _ => false,
            }
        })
    }

    /// Every triangulation, by a subset search over the full-dimensional
    /// simplices in decreasing order of volume, pruned by volume.
    fn all_triangulations(&self, total: i128) -> BTreeSet<Labels> {
        let mut simplices: Vec<(i128, Vec<usize>)> = combinations(self.rows.len(), self.rank)
            .map(|s| {
                let v = s.to_vec();
                (self.det_of(&v).abs(), v)
            })
            .filter(|(vol, _)| *vol != 0)
            .collect();
        simplices.sort_by_key(|s| std::cmp::Reverse(s.0));
        let mut suffix = vec![0; simplices.len() + 1];
        for i in (0..simplices.len()).rev() {
            suffix[i] = suffix[i + 1] + simplices[i].0;
        }
        let mut search = Search {
            points: self,
            total,
            simplices: &simplices,
            suffix: &suffix,
            chosen: Vec::new(),
            out: BTreeSet::new(),
        };
        search.run(0, 0);
        search.out
    }
}

struct Search<'a> {
    points: &'a Points,
    total: i128,
    simplices: &'a [(i128, Vec<usize>)],
    suffix: &'a [i128],
    chosen: Labels,
    out: BTreeSet<Labels>,
}

impl Search<'_> {
    fn run(&mut self, i: usize, acc: i128) {
        if acc == self.total {
            if self.points.is_triangulation(&self.chosen, self.total) {
                let mut c = self.chosen.clone();
                c.sort();
                self.out.insert(c);
            }
            return;
        }
        if i == self.simplices.len() || acc + self.suffix[i] < self.total {
            return;
        }
        let (vol, ref s) = self.simplices[i];
        if acc + vol <= self.total {
            self.chosen.push(s.clone());
            self.run(i + 1, acc + vol);
            self.chosen.pop();
        }
        self.run(i + 1, acc);
    }
}

fn random_rows(rng: &mut Rng, n: usize, d: usize, bound: i64) -> Vec<Vec<i64>> {
    let mut rows: Vec<Vec<i64>> = Vec::new();
    while rows.len() < n {
        if !rows.is_empty() && rng.below(10) == 0 {
            let k = rng.below(rows.len() as u64) as usize;
            rows.push(rows[k].clone());
            continue;
        }
        rows.push((0..d).map(|_| rng.signed(bound)).collect());
    }
    rows
}

fn as_set(ts: &[Triangulation]) -> BTreeSet<Labels> {
    ts.iter().map(Triangulation::to_vecs).collect()
}

/// Run `cases` random configurations; returns how many were full rank and
/// compared.
fn run(seed: u64, cases: usize) -> usize {
    let mut rng = Rng(seed);
    let mut compared = 0;
    for case in 0..cases {
        let d = 2 + rng.below(2) as usize;
        let n = d + 1 + rng.below(if d == 2 { 5 } else { 3 }) as usize;
        let bound = [1, 1, 2][rng.below(3) as usize];
        let rows = random_rows(&mut rng, n, d, bound);
        let what = format!("case {case}: {rows:?}");
        let c = PointConfig::from_row_major(&rows.concat(), n, d).unwrap();
        // Rank-deficient configurations are rejected; that is tested elsewhere.
        let Ok(chi) = c.chirotope() else { continue };
        let points = Points {
            rows: rows
                .iter()
                .map(|r| r.iter().map(|&x| i128::from(x)).chain([1]).collect())
                .collect(),
            rank: d + 1,
        };
        let placing = triang::placing(&chi).unwrap();
        let total: i128 = placing
            .to_vecs()
            .iter()
            .map(|s| points.det_of(s).abs())
            .sum();
        let want = points.all_triangulations(total);
        let want_fine: BTreeSet<Labels> = want
            .iter()
            .filter(|t| t.iter().flatten().collect::<BTreeSet<_>>().len() == n)
            .cloned()
            .collect();

        for w in &want {
            let set: SimplexSet = w.iter().map(|s| Simplex::from(&s[..])).collect();
            let t = Triangulation::new(n, d + 1, set).unwrap();
            assert!(t.check(&chi).is_ok(), "{what}: check rejects {w:?}");
        }

        let all = triang::all_triangulations(&chi, false).unwrap();
        assert_eq!(as_set(&all), want, "{what}: all_triangulations");
        let fine = triang::all_triangulations(&chi, true).unwrap();
        assert_eq!(as_set(&fine), want_fine, "{what}: all_triangulations fine");

        let connected = triang::all_connected(&chi, false).unwrap();
        assert_eq!(as_set(&connected), want, "{what}: all_connected");
        for t in &connected {
            assert!(
                points.is_triangulation(&t.to_vecs(), total),
                "{what}: invalid {:?}",
                t.to_vecs()
            );
            for nb in triang::neighbors(&chi, t).unwrap() {
                assert!(
                    triang::neighbors(&chi, &nb).unwrap().contains(t),
                    "{what}: asymmetric {:?} -> {:?}",
                    t.to_vecs(),
                    nb.to_vecs()
                );
            }
        }
        let connected_fine = as_set(&triang::all_connected(&chi, true).unwrap());
        assert!(connected_fine.is_subset(&want_fine), "{what}: only_fine");
        if d == 2 {
            assert_eq!(connected_fine, want_fine, "{what}: only_fine in the plane");
        }
        compared += 1;
    }
    compared
}

#[test]
fn enumerations_match_brute_force() {
    assert!(run(2024, 25) > 15);
}

#[test]
#[ignore = "600 configurations; run in release"]
fn enumerations_match_brute_force_long() {
    assert!(run(2025, 600) > 400);
}
