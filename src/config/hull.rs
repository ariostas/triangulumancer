//! Exact reduction of a configuration to its own affine (or linear) hull.
//!
//! A configuration that does not span its ambient space is re-expressed, with
//! the same labels in the same order, as a full-rank configuration of lower
//! dimension. This replaces CGAL's incremental dimension growth (all of
//! `Coaffine.h` and `insert_increase_dimension`, `CGAL.md` section 6.3), and is
//! what fixes bug B4 structurally: a rank-deficient configuration is
//! triangulated in its own hull rather than producing a padded, misshapen
//! simplices array.
//!
//! # The construction
//!
//! For a point configuration, take the differences `p_i - p_0`; for a vector
//! configuration, the vectors themselves. These generate a lattice `L` of rank
//! `s`. Its column-style Hermite normal form `H` (an `ambient x s` integer
//! matrix in echelon form) is a basis of `L`, and every generator has unique
//! *integer* coordinates `c_i` with `H c_i = g_i`. The reduced configuration is
//! the `c_i`, which is full-rank in dimension `s`.
//!
//! The map `c -> H c` is an injective linear map onto the span, so the
//! reduction preserves every affine (or linear) dependency, and hence the
//! oriented matroid up to one global sign. It also preserves the integer
//! structure: the reduced points generate `Z^s` as an affine lattice, so the
//! normalised volume of a simplex is its volume relative to the lattice the
//! configuration itself generates.
//!
//! It does **not** preserve Euclidean distances. Anything metric must be
//! computed in the original coordinates and carried over by label; in
//! particular a regular triangulation should lift each point by its height
//! computed from the *original* coordinates (`|p|^2 - w` for weights), then
//! triangulate the reduced points with those heights. That is sound because
//! the regular subdivision induced by given heights is invariant under affine
//! isomorphisms of the points.
//!
//! All arithmetic is over arbitrary-precision integers. This runs once per
//! configuration, so its speed does not matter.

use dashu_int::IBig;
use dashu_int::ops::DivEuclid;

use super::{Config, Kind};
use crate::error::{Error, Result};

impl<K: Kind> Config<K> {
    /// The configuration re-expressed in a lattice basis of its own affine
    /// hull (points) or linear span (vectors).
    ///
    /// The result has the same labels in the same order, is full rank, and has
    /// dimension `rank - 1` for points or `rank` for vectors. A configuration
    /// that is already full rank is returned unchanged, coordinates included.
    /// See the module documentation for what is and is not preserved.
    ///
    /// # Errors
    ///
    /// [`Error::EmptyConfiguration`] if there are no points or vectors, and
    /// [`Error::Overflow`] if a reduced coordinate does not fit in `i64`. With
    /// the Hermite basis reduced, a coordinate is bounded by about
    /// `2^(rank - 1)` times the largest input difference, so the latter needs
    /// coordinates near the top of the `i64` range.
    pub fn reduce_to_hull(&self) -> Result<Self> {
        if self.is_empty() {
            return Err(Error::EmptyConfiguration);
        }
        if self.is_full_rank() {
            return Ok(self.clone());
        }

        let dim = self.dim();
        let origin: Vec<IBig> = match K::KIND {
            crate::ConfigKind::Point => self
                .iter()
                .next()
                .unwrap_or(&[])
                .iter()
                .map(|&x| IBig::from(x))
                .collect(),
            _ => vec![IBig::ZERO; dim],
        };
        let generators: Vec<Vec<IBig>> = self
            .iter()
            .map(|p| {
                p.iter()
                    .zip(&origin)
                    .map(|(&x, o)| IBig::from(x) - o)
                    .collect()
            })
            .collect();

        let (basis, pivots) = hermite_basis(generators.clone(), dim);
        let s = basis.len();
        let expected = self.rank() - usize::from(K::KIND.is_point());
        if s != expected {
            return Err(Error::Internal("affine-hull reduction: lattice rank"));
        }

        let mut coords = Vec::with_capacity(self.len() * s);
        for g in &generators {
            for c in coordinates(&basis, &pivots, g)? {
                coords.push(
                    i64::try_from(&c).map_err(|_| Error::Overflow("affine-hull coordinates"))?,
                );
            }
        }
        let reduced = Self::from_row_major(&coords, self.len(), s)?;
        if !reduced.is_full_rank() {
            return Err(Error::Internal(
                "affine-hull reduction: result not full rank",
            ));
        }
        Ok(reduced)
    }
}

/// `(g, x, y)` with `g = gcd(a, b) >= 0` and `a x + b y = g`.
fn extended_gcd(a: &IBig, b: &IBig) -> (IBig, IBig, IBig) {
    let (mut r0, mut r1) = (a.clone(), b.clone());
    let (mut s0, mut s1) = (IBig::ONE, IBig::ZERO);
    let (mut t0, mut t1) = (IBig::ZERO, IBig::ONE);
    while !r1.is_zero() {
        let q = &r0 / &r1;
        let r2 = &r0 - &q * &r1;
        let s2 = &s0 - &q * &s1;
        let t2 = &t0 - &q * &t1;
        (r0, r1) = (r1, r2);
        (s0, s1) = (s1, s2);
        (t0, t1) = (t1, t2);
    }
    if r0 < IBig::ZERO {
        (-r0, -s0, -t0)
    } else {
        (r0, s0, t0)
    }
}

/// The column-style Hermite normal form of the generators (each of length
/// `rows`): a basis of the lattice they generate, as columns in echelon form,
/// together with the pivot row of each basis column.
///
/// Column `t` of the basis is zero above row `pivots[t]`, positive there, and
/// every earlier basis column's entry in that row is reduced into
/// `0..pivot`. Only unimodular column operations are used, so the lattice is
/// unchanged.
fn hermite_basis(mut cols: Vec<Vec<IBig>>, rows: usize) -> (Vec<Vec<IBig>>, Vec<usize>) {
    let mut pivots = Vec::new();
    let mut p = 0;
    for row in 0..rows {
        if p == cols.len() {
            break;
        }
        // Fold every later column into column p, gcd-style, until only
        // column p has a non-zero entry in this row.
        for j in p + 1..cols.len() {
            if cols[j][row].is_zero() {
                continue;
            }
            let (a, b) = (cols[p][row].clone(), cols[j][row].clone());
            let (g, x, y) = extended_gcd(&a, &b);
            let (a_g, b_g) = (&a / &g, &b / &g);
            // [col_p, col_j] <- [x col_p + y col_j, a/g col_j - b/g col_p],
            // a unimodular change of basis (determinant x a/g + y b/g = 1).
            let (cp, cj) = (&cols[p], &cols[j]);
            let new_p: Vec<IBig> = cp.iter().zip(cj).map(|(u, v)| &x * u + &y * v).collect();
            let new_j: Vec<IBig> = cp
                .iter()
                .zip(cj)
                .map(|(u, v)| &a_g * v - &b_g * u)
                .collect();
            cols[p] = new_p;
            cols[j] = new_j;
        }
        if cols[p][row].is_zero() {
            continue;
        }
        if cols[p][row] < IBig::ZERO {
            cols[p].iter_mut().for_each(|x| *x = -std::mem::take(x));
        }
        // Reduce the earlier basis columns in this row modulo the pivot, which
        // keeps the entries (and so the coordinates) small.
        let pivot = cols[p][row].clone();
        for q in 0..p {
            let f = (&cols[q][row]).div_euclid(&pivot);
            if !f.is_zero() {
                let cp = cols[p].clone();
                cols[q].iter_mut().zip(&cp).for_each(|(x, y)| *x -= &f * y);
            }
        }
        pivots.push(row);
        p += 1;
    }
    cols.truncate(p);
    (cols, pivots)
}

/// The integer coordinates `c` of `g` in the basis: `basis * c = g`, by
/// forward substitution on the pivot rows.
fn coordinates(basis: &[Vec<IBig>], pivots: &[usize], g: &[IBig]) -> Result<Vec<IBig>> {
    let mut c: Vec<IBig> = Vec::with_capacity(basis.len());
    for (t, &row) in pivots.iter().enumerate() {
        let mut rhs = g[row].clone();
        for (q, cq) in c.iter().enumerate() {
            rhs -= &basis[q][row] * cq;
        }
        let pivot = &basis[t][row];
        let (quot, rem) = (&rhs / pivot, &rhs % pivot);
        if !rem.is_zero() {
            return Err(Error::Internal(
                "affine-hull reduction: point not in lattice",
            ));
        }
        c.push(quot);
    }
    // The pivot rows determine `c`; the other rows must agree.
    for (row, gr) in g.iter().enumerate() {
        let mut acc = IBig::ZERO;
        for (b, cq) in basis.iter().zip(&c) {
            acc += &b[row] * cq;
        }
        if acc != *gr {
            return Err(Error::Internal("affine-hull reduction: point not in span"));
        }
    }
    Ok(c)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arith::Sign;
    use crate::config::{PointConfig, VectorConfig};
    use crate::simplex::Simplex;

    fn pc(coords: &[&[i64]]) -> PointConfig {
        let mut c = PointConfig::new();
        for p in coords {
            c.push(p).unwrap();
        }
        c
    }

    #[test]
    fn collinear_points_reduce_to_a_line() {
        // `rank_deficient_collinear` from the fixture corpus (bug B4).
        let p = pc(&[&[0, 0], &[1, 1], &[2, 2], &[3, 3]]);
        let r = p.reduce_to_hull().unwrap();
        assert_eq!(r.dim(), 1);
        assert_eq!(r.to_row_major(), vec![0, 1, 2, 3]);
        assert!(r.is_full_rank());
    }

    #[test]
    fn coplanar_points_reduce_to_a_plane() {
        // `rank_deficient_coplanar_3d`: the unit square in z = 0.
        let p = pc(&[&[0, 0, 0], &[1, 0, 0], &[0, 1, 0], &[1, 1, 0]]);
        let r = p.reduce_to_hull().unwrap();
        assert_eq!(r.dim(), 2);
        assert_eq!(r.len(), 4);
        assert!(r.is_full_rank());
        // Still a unimodular square: every triangle has volume +-1.
        for t in [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]] {
            assert_ne!(r.sign_of_simplex(&Simplex::from(t)).unwrap(), Sign::Zero);
        }
    }

    #[test]
    fn single_point_reduces_to_dimension_zero() {
        let r = pc(&[&[5, -7]]).reduce_to_hull().unwrap();
        assert_eq!((r.dim(), r.len()), (0, 1));
        assert!(r.is_full_rank());
        // Duplicates of one point too.
        let r = pc(&[&[5, -7], &[5, -7]]).reduce_to_hull().unwrap();
        assert_eq!((r.dim(), r.len()), (0, 2));
    }

    #[test]
    fn vectors_reduce_to_their_span() {
        let v = VectorConfig::from_row_major(&[2, 2, 4, 4, -2, -2], 3, 2).unwrap();
        let r = v.reduce_to_hull().unwrap();
        assert_eq!(r.dim(), 1);
        assert_eq!(r.to_row_major(), vec![1, 2, -1]);
        let zero = VectorConfig::from_row_major(&[0, 0, 0, 0], 2, 2).unwrap();
        let r = zero.reduce_to_hull().unwrap();
        assert_eq!((r.dim(), r.len()), (0, 2));
    }

    #[test]
    fn full_rank_is_unchanged_and_empty_is_an_error() {
        let p = pc(&[&[0, 0], &[3, 0], &[0, 5]]);
        assert_eq!(p.reduce_to_hull().unwrap(), p);
        assert_eq!(
            PointConfig::new().reduce_to_hull(),
            Err(Error::EmptyConfiguration)
        );
    }

    #[test]
    fn extreme_coordinates() {
        // Differences of 2^63 do not fit in i64, but the lattice they
        // generate has a small basis.
        let big = 1i64 << 62;
        let p = pc(&[&[-big, big], &[big, -big], &[0, 0]]);
        let r = p.reduce_to_hull().unwrap();
        assert_eq!(r.to_row_major(), vec![0, 2, 1]);
        // Here the lattice is Z itself and the span is 2^64 - 1: an error,
        // not a panic or a wrapped value.
        let p = pc(&[&[i64::MIN, 0], &[i64::MAX, 0], &[0, 0]]);
        assert_eq!(
            p.reduce_to_hull(),
            Err(Error::Overflow("affine-hull coordinates"))
        );
    }

    #[test]
    fn hermite_basis_is_echelon_and_reduced() {
        let gens: Vec<Vec<IBig>> = [[4, 6, 1], [6, 9, 2], [2, 3, 0]]
            .iter()
            .map(|g| g.iter().map(|&x| IBig::from(x)).collect())
            .collect();
        let (basis, pivots) = hermite_basis(gens, 3);
        assert_eq!(basis.len(), 2);
        assert_eq!(pivots, vec![0, 2]);
        for (t, &row) in pivots.iter().enumerate() {
            assert!(basis[t][row] > IBig::ZERO);
            for b in &basis[t + 1..] {
                assert!(b[row].is_zero());
            }
            for b in &basis[..t] {
                assert!(b[row] >= IBig::ZERO && b[row] < basis[t][row]);
            }
        }
    }

    /// A tiny deterministic generator for the embedding test.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self, bound: i64) -> i64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 33) % (2 * bound as u64 + 1)) as i64 - bound
        }
    }

    /// Embed a full-rank configuration `Q` in `Z^s` into `Z^d` by an injective
    /// integer affine map. The reduction must recover the oriented matroid of
    /// `Q`, up to one global sign.
    #[test]
    fn reduction_preserves_the_oriented_matroid() {
        let mut rng = Lcg(7);
        let mut checked = 0;
        for trial in 0..200 {
            let s = 1 + trial % 3;
            let d = s + 1 + trial % 2;
            let n = s + 2 + trial % 3;
            let q: Vec<i64> = (0..n * s).map(|_| rng.next(4)).collect();
            let q = PointConfig::from_row_major(&q, n, s).unwrap();
            if !q.is_full_rank() {
                continue;
            }
            let a: Vec<i64> = (0..d * s).map(|_| rng.next(5)).collect();
            let t: Vec<i64> = (0..d).map(|_| rng.next(50)).collect();
            let mut emb = Vec::with_capacity(n * d);
            for p in q.iter() {
                for i in 0..d {
                    emb.push(t[i] + (0..s).map(|k| a[i * s + k] * p[k]).sum::<i64>());
                }
            }
            let p = PointConfig::from_row_major(&emb, n, d).unwrap();
            if p.rank() != s + 1 {
                continue; // the random map was not injective
            }
            let r = p.reduce_to_hull().unwrap();
            assert_eq!(r.dim(), s);
            let mut global: Option<Sign> = None;
            for mask in 0u32..(1 << n) {
                if mask.count_ones() as usize != s + 1 {
                    continue;
                }
                let simplex: Simplex = (0..n).filter(|&i| mask & (1 << i) != 0).collect();
                let (sq, sr) = (
                    q.sign_of_simplex(&simplex).unwrap(),
                    r.sign_of_simplex(&simplex).unwrap(),
                );
                assert_eq!(sq.is_zero(), sr.is_zero());
                if !sq.is_zero() {
                    let rel = sq * sr;
                    assert_eq!(*global.get_or_insert(rel), rel);
                }
            }
            checked += 1;
        }
        assert!(checked > 100);
    }
}
