//! Regular and Delaunay triangulations: the CGAL replacement.
//!
//! The regular triangulation of points `pᵢ` with lifts `ℓᵢ` is the vertical
//! projection of the lower faces of the convex hull of the lifted points
//! `(pᵢ, ℓᵢ)` (`CGAL.md` section 2.2). The three entry points differ only in
//! the lifts:
//!
//! * [`with_heights`] — the heights *are* the lifts;
//! * [`with_weights`] — `ℓᵢ = |pᵢ|² − wᵢ` (weighted Delaunay, CGAL's
//!   convention);
//! * [`delaunay`] — `ℓᵢ = |pᵢ|²`.
//!
//! The implementation follows `CGAL.md` section 6.6: CGAL's incremental
//! algorithm and data structure (`insert.rs`, `tds.rs`) on a single exact
//! determinant predicate (`predicates.rs`), after an exact reduction of the
//! configuration to its own affine hull. Lifts are exact ([`lift`]).
//!
//! # What is returned
//!
//! A [`Triangulation`] of the configuration whose simplices have
//! `dim(aff P) + 1` labels — fewer than `dim + 1` for a configuration that
//! does not span its ambient space, which 0.3 turned into a malformed array
//! (bug B4). Points that are not vertices are the *hidden* points, and are
//! exactly [`Triangulation::unused`]: a point whose lift lies above the lower
//! hull, or a duplicate that lost the tie (the lowest lift wins, then the
//! lowest index).
//!
//! # Ties
//!
//! When several lifted points lie on a common lower facet the regular
//! subdivision is not a triangulation, and one is chosen by CGAL's symbolic
//! perturbation, so the result agrees with 0.3 on degenerate input too. The
//! rule: perturb the lifts upwards by `ε^π(i)`, where the lexicographically
//! largest point (by original coordinates, then by label) gets the largest
//! raise, the next largest the next, and so on. The triangulation is then the
//! unique regular triangulation of the perturbed lifts, a deterministic
//! function of the points and lifts that the insertion order does not affect.
//! `src/regular/predicates.rs` has the details.
//!
//! This is the one module besides [`config`](crate::config) that uses
//! coordinates: a regular triangulation is not determined by the oriented
//! matroid (`PLAN.md` section 6.1).

mod insert;
pub mod lift;
mod predicates;
mod tds;

use dashu_int::IBig;

pub use lift::Scalar;

use crate::arith;
use crate::config::PointConfig;
use crate::error::{Error, Result};
use crate::simplex::{Simplex, SimplexSet};
use crate::triang::Triangulation;

/// The Delaunay triangulation: lifts `|pᵢ|²`.
///
/// # Errors
///
/// [`Error::EmptyConfiguration`] for a configuration without points; see
/// [`from_lifts`] for the rest.
pub fn delaunay(config: &PointConfig) -> Result<Triangulation> {
    from_lifts(config, lift::delaunay(config))
}

/// The regular triangulation for weights `wᵢ`: lifts `|pᵢ|² − wᵢ`.
///
/// # Errors
///
/// [`Error::ParameterCountMismatch`] unless there is one weight per point,
/// [`Error::InvalidParameter`] for an unacceptable weight (see
/// [`lift::from_weights`]), and as [`delaunay`].
pub fn with_weights(config: &PointConfig, weights: &[Scalar]) -> Result<Triangulation> {
    from_lifts(config, lift::from_weights(config, weights)?)
}

/// The regular triangulation for heights `hᵢ`: lifts `hᵢ`.
///
/// # Errors
///
/// As [`with_weights`].
pub fn with_heights(config: &PointConfig, heights: &[Scalar]) -> Result<Triangulation> {
    from_lifts(config, lift::from_heights(config, heights)?)
}

/// The regular triangulation for integer lifts `ℓᵢ`.
///
/// # Errors
///
/// [`Error::EmptyConfiguration`] for a configuration without points,
/// [`Error::ParameterCountMismatch`] unless there is one lift per point,
/// [`Error::Overflow`] if the affine-hull reduction leaves the `i64` range or
/// the configuration has more than `2^32 - 1` points or dimension above 254,
/// and [`Error::Internal`] if an internal invariant fails (a bug).
pub fn from_lifts(config: &PointConfig, lifts: Vec<IBig>) -> Result<Triangulation> {
    let n = config.len();
    if n == 0 {
        return Err(Error::EmptyConfiguration);
    }
    if lifts.len() != n {
        return Err(Error::ParameterCountMismatch {
            expected: n,
            found: lifts.len(),
        });
    }
    if n >= u32::MAX as usize {
        return Err(Error::Overflow("number of points"));
    }
    let reduced = config.reduce_to_hull()?;
    let d = reduced.dim();
    if d > 254 {
        return Err(Error::Overflow("dimension"));
    }
    let geometry = predicates::Geometry::new(&reduced, config, lifts);

    let cells: Vec<Vec<u32>> = if d == 0 {
        // Every point is the same point; the lowest lift is the only vertex.
        geometry.lowest().into_iter().map(|v| vec![v]).collect()
    } else {
        let initial = initial_simplex(&reduced)?;
        let mut builder = insert::Builder::new(&geometry, &initial)?;
        for p in insertion_order(&reduced, &initial) {
            builder.insert(p)?;
        }
        builder.finish().0
    };
    let simplices: SimplexSet = cells
        .iter()
        .map(|c| c.iter().map(|&v| v as usize).collect::<Simplex>())
        .collect();
    Triangulation::new(n, d + 1, simplices)
}

/// The order to insert every point but `initial` in: a biased randomised
/// insertion order (BRIO) with each round sorted along a Z-order curve, as
/// CGAL's range `insert` does with its Hilbert sort.
///
/// The result does not depend on the order — it is the unique regular
/// triangulation of the symbolically perturbed lifts (see
/// [`predicates`]) — but the cost does: in input order every point-location
/// walk crosses a constant fraction of the triangulation, while along a
/// space-filling curve it starts next to where the point lands. The random
/// rounds (each holding about half the points not yet inserted) keep the
/// triangulation well shaped for the curve to work on, which a single sorted
/// pass does not guarantee. The shuffle is seeded, so the work done is
/// deterministic.
fn insertion_order(reduced: &PointConfig, initial: &[u32]) -> Vec<u32> {
    let n = reduced.len();
    let d = reduced.dim();
    let mut order: Vec<u32> = (0..n as u32).filter(|p| !initial.contains(p)).collect();
    // A seeded Fisher-Yates shuffle (xorshift64).
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    for i in (1..order.len()).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        order.swap(i, (state % (i as u64 + 1)) as usize);
    }
    let bits_per_dim = 128 / d.max(1);
    if bits_per_dim == 0 || order.len() < 64 {
        return order;
    }
    // Offsets from the minimum fit in a u64 whatever the coordinates.
    let mut min = vec![i64::MAX; d];
    for j in 0..n {
        for (m, &x) in min.iter_mut().zip(reduced.coords(j).unwrap_or_default()) {
            *m = (*m).min(x);
        }
    }
    let offset = |j: u32, k: usize| {
        let x = reduced.coords(j as usize).map_or(0, |c| c[k]);
        (i128::from(x) - i128::from(min[k])) as u64
    };
    let width = (0..n as u32)
        .flat_map(|j| (0..d).map(move |k| (j, k)))
        .map(|(j, k)| 64 - offset(j, k).leading_zeros() as usize)
        .max()
        .unwrap_or(0);
    let bits = bits_per_dim.min(width);
    let shift = width - bits;
    let key = |j: u32| {
        let mut key = 0u128;
        for b in (0..bits).rev() {
            for k in 0..d {
                key = (key << 1) | u128::from((offset(j, k) >> (shift + b)) & 1);
            }
        }
        key
    };
    // Rounds from the back: the last half, then half of the rest, and so on;
    // the first round takes what is left once a round would be small.
    let mut end = order.len();
    while end > 0 {
        let start = if end <= 64 { 0 } else { end / 2 };
        order[start..end].sort_by_cached_key(|&j| key(j));
        end = start;
    }
    order
}

/// The first `dim + 1` affinely independent points, by greedy rank extension
/// in label order.
fn initial_simplex(reduced: &PointConfig) -> Result<Vec<u32>> {
    let rows = reduced.rows();
    let mut chosen: Vec<u32> = Vec::with_capacity(rows);
    let mut columns: Vec<i64> = Vec::with_capacity(rows * rows);
    for j in 0..reduced.len() {
        let col = reduced
            .column(j)
            .ok_or(Error::Internal("initial simplex"))?;
        columns.extend_from_slice(col);
        if arith::rank(&columns, chosen.len() + 1, rows)? == chosen.len() + 1 {
            chosen.push(j as u32);
            if chosen.len() == rows {
                return Ok(chosen);
            }
        } else {
            columns.truncate(columns.len() - rows);
        }
    }
    Err(Error::Internal(
        "initial simplex: the reduced configuration is not full rank",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pc(rows: &[&[i64]]) -> PointConfig {
        let mut c = PointConfig::new();
        for r in rows {
            c.push(r).unwrap();
        }
        c
    }

    fn ints(xs: &[i64]) -> Vec<Scalar> {
        xs.iter().map(|&x| x.into()).collect()
    }

    #[test]
    fn small_cases() {
        let tri = pc(&[&[0, 0], &[1, 0], &[0, 1]]);
        assert_eq!(delaunay(&tri).unwrap().to_vecs(), vec![vec![0, 1, 2]]);

        let square = pc(&[&[0, 0], &[1, 0], &[1, 1], &[0, 1]]);
        assert_eq!(
            delaunay(&square).unwrap().to_vecs(),
            vec![vec![0, 1, 3], vec![1, 2, 3]]
        );
        let t = with_heights(&square, &ints(&[0, 0, 5, 0])).unwrap();
        assert_eq!(t.to_vecs(), vec![vec![0, 1, 3], vec![1, 2, 3]]);
        let t = with_heights(&square, &ints(&[0, 5, 0, 0])).unwrap();
        assert_eq!(t.to_vecs(), vec![vec![0, 1, 2], vec![0, 2, 3]]);

        // An interior point lifted high is hidden.
        let p = pc(&[&[0, 0], &[3, 0], &[0, 3], &[1, 1]]);
        let t = with_heights(&p, &ints(&[0, 0, 0, 1])).unwrap();
        assert_eq!(t.to_vecs(), vec![vec![0, 1, 2]]);
        assert_eq!(t.unused(), Simplex::from([3]));
        let t = with_heights(&p, &ints(&[0, 0, 0, -1])).unwrap();
        assert_eq!(t.len(), 3);
        assert!(t.is_fine());
    }

    #[test]
    fn degenerate_inputs() {
        // B4: collinear points triangulate in their own line.
        let line = pc(&[&[0, 0], &[1, 1], &[2, 2], &[3, 3]]);
        let t = delaunay(&line).unwrap();
        assert_eq!(t.to_vecs(), vec![vec![0, 1], vec![1, 2], vec![2, 3]]);
        assert_eq!(t.rank(), 2);
        let t = delaunay(&pc(&[&[0, 0]])).unwrap();
        assert_eq!(t.to_vecs(), vec![vec![0]]);
        // Every point the same: the lowest lift, then the lowest index, wins.
        let same = pc(&[&[1, 2], &[1, 2], &[1, 2]]);
        assert_eq!(delaunay(&same).unwrap().to_vecs(), vec![vec![0]]);
        let t = with_heights(&same, &ints(&[3, 1, 1])).unwrap();
        assert_eq!(t.to_vecs(), vec![vec![1]]);
        // B5: duplicates keep the lowest index.
        let dup = pc(&[&[0, 0], &[1, 0], &[0, 1], &[1, 0]]);
        let t = delaunay(&dup).unwrap();
        assert_eq!(t.to_vecs(), vec![vec![0, 1, 2]]);
        assert_eq!(t.unused(), Simplex::from([3]));
        let t = with_heights(&dup, &ints(&[0, 1, 0, 0])).unwrap();
        assert_eq!(t.to_vecs(), vec![vec![0, 2, 3]]);

        assert_eq!(
            delaunay(&PointConfig::new()),
            Err(Error::EmptyConfiguration)
        );
        assert!(matches!(
            with_weights(&dup, &ints(&[1, 2])),
            Err(Error::ParameterCountMismatch { .. })
        ));
    }
}
