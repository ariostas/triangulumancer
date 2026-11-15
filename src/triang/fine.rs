//! The fine triangulation: the placing triangulation with every unused element
//! flipped in.
//!
//! A port of TOPCOM's `FineTriang` (`lib-src/FineTriang.cc`; `TOPCOM.md`
//! section 4), including the fork's patch for degenerate circuits.

use super::{Triangulation, placing};
use crate::arith::Sign;
use crate::error::Result;
use crate::om::{Circuit, OrientedMatroid};
use crate::simplex::{Simplex, SimplexSet};

/// The fine triangulation of `om`.
///
/// Start from the [`placing`] triangulation. Then, for each element `i` it
/// does not use, in increasing order, find a simplex `S` such that the circuit
/// in `S ∪ {i}` has `i` alone on one side — so `i` lies in the relative
/// interior of the face `σ` formed by the other side — and replace the star of
/// `σ` by its stellar subdivision at `i`.
///
/// The name is TOPCOM's, and so is the result, simplex for simplex — which is
/// **not always fine**. An element that no simplex accepts (a zero vector, for
/// instance) is left out, and a duplicate of a used point *replaces* that
/// point rather than joining it (the two are each other's circuit), so with
/// repeated points the result uses only the last copy of each.
///
/// TOPCOM scans the simplices in the order of its global intern table and
/// flips at the first match; this scans them in canonical order. The two agree
/// because the face `σ` is unique: the relative interiors of the faces of a
/// triangulation are disjoint, and the flip depends only on `σ`.
///
/// # Errors
///
/// As [`placing`], and whatever [`OrientedMatroid::sign`] returns.
pub fn fine(om: &impl OrientedMatroid) -> Result<Triangulation> {
    let start = placing(om)?;
    let (n, rank) = (start.n(), start.rank());
    let unused = start.unused();
    let mut t = start.into_simplices();
    for i in &unused {
        flip_in(om, &mut t, i)?;
    }
    Triangulation::new(n, rank, t)
}

/// Flip `i` into `t`, if some simplex of `t` accepts it.
fn flip_in(om: &impl OrientedMatroid, t: &mut SimplexSet, i: usize) -> Result<()> {
    let mut found = None;
    for s in t.iter() {
        let mut dependent = s.clone();
        dependent.insert(i);
        if !alone_in_circuit(om, &dependent, i)? {
            continue;
        }
        let z = Circuit::of_dependent_set(om, &dependent)?;
        // `i` alone on one side: the other side is the face it subdivides.
        // The circuit's own orientation is arbitrary, so try both; a circuit
        // whose other side is empty (a positive dependency, possible for
        // vectors) has nothing to subdivide, and TOPCOM's fork skips it.
        let lone = Simplex::from([i]);
        if z.positive() == &lone && !z.negative().is_empty() {
            found = Some(z);
            break;
        }
        if z.negative() == &lone && !z.positive().is_empty() {
            found = Some(z.negated());
            break;
        }
    }
    let Some(z) = found else {
        return Ok(());
    };
    // With `Z⁺ = {i}`: the lower facets are the single face `Z⁻`, and the
    // upper facets its subdivision by `i`. Both are joined with the link of
    // that face to replace its star.
    let lower = z.lower_facets();
    let upper = z.upper_facets();
    let face = lower.iter().next().cloned().unwrap_or_else(Simplex::new);
    let link = t.link(&face);
    for s in &lower.join(&link) {
        t.remove(s);
    }
    t.extend(upper.join(&link));
    Ok(())
}

/// Whether `i` is alone on one side of the circuit in `dependent` (a basis
/// plus `i`) with the other side non-empty: the test below on
/// [`Circuit::of_dependent_set`], decided from the same coefficients but
/// stopping at the first one on `i`'s side, so that the simplices not
/// containing `i` — nearly all of them — usually cost one or two signs
/// rather than `rank + 1`.
fn alone_in_circuit(om: &impl OrientedMatroid, dependent: &Simplex, i: usize) -> Result<bool> {
    let coefficient = |pos: usize, e: usize| -> Result<Sign> {
        let mut basis = dependent.clone();
        basis.remove(e);
        let chi = om.sign(&basis)?;
        Ok(if pos % 2 == 0 { chi } else { -chi })
    };
    let pos_i = dependent.iter().position(|e| e == i).unwrap_or(0);
    let side = coefficient(pos_i, i)?;
    if side.is_zero() {
        return Ok(false);
    }
    let mut other = false;
    for (pos, e) in dependent.iter().enumerate() {
        if e == i {
            continue;
        }
        let c = coefficient(pos, e)?;
        if c == side {
            return Ok(false);
        }
        other |= !c.is_zero();
    }
    Ok(other)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{PointConfig, VectorConfig};

    #[test]
    fn interior_point_is_coned() {
        let p = PointConfig::from_row_major(&[0, 0, 3, 0, 0, 3, 1, 1], 4, 2).unwrap();
        let chi = p.chirotope().unwrap();
        let t = fine(&chi).unwrap();
        assert_eq!(
            t.to_vecs(),
            vec![vec![0, 1, 3], vec![0, 2, 3], vec![1, 2, 3]]
        );
        assert!(t.is_fine());
        assert_eq!(t.check(&chi), Ok(()));
    }

    #[test]
    fn point_on_an_interior_edge() {
        // 4 is the midpoint of the diagonal 0-2 of the square.
        let p = PointConfig::from_row_major(&[0, 0, 2, 0, 2, 2, 0, 2, 1, 1], 5, 2).unwrap();
        let chi = p.chirotope().unwrap();
        let t = fine(&chi).unwrap();
        assert_eq!(t.len(), 4);
        assert!(t.is_fine());
        assert_eq!(t.check(&chi), Ok(()));
    }

    #[test]
    fn duplicates_replace_rather_than_join() {
        let p = PointConfig::from_row_major(&[0, 0, 1, 0, 0, 1, 1, 0], 4, 2).unwrap();
        let chi = p.chirotope().unwrap();
        let t = fine(&chi).unwrap();
        assert_eq!(t.to_vecs(), vec![vec![0, 2, 3]]);
        assert!(!t.is_fine());
    }

    #[test]
    fn zero_vectors_are_left_out() {
        let v = VectorConfig::from_row_major(&[1, 0, 0, 1, 0, 0, -1, -1], 4, 2).unwrap();
        let chi = v.chirotope().unwrap();
        let t = fine(&chi).unwrap();
        assert_eq!(t.unused(), Simplex::from([2]));
        assert_eq!(t.check(&chi), Ok(()));
    }
}
