//! The placing (pushing) triangulation.
//!
//! A port of TOPCOM's `PlacingTriang` (`lib-src/PlacingTriang.cc`;
//! `TOPCOM.md` section 3): beneath–beyond on the chirotope alone.

use super::Triangulation;
use crate::error::{Error, Result};
use crate::hash::FastHashMap;
use crate::om::OrientedMatroid;
use crate::simplex::Simplex;

/// The placing triangulation of `om`.
///
/// Start from the lexicographically first basis
/// ([`OrientedMatroid::first_basis`]), then place the remaining elements in
/// increasing order: each new element `i` is coned to every boundary facet of
/// the current triangulation that it sees, i.e. that has `i` strictly on the
/// other side from the simplex behind it. An element that sees no facet — one
/// inside the hull, or cone, of the elements placed before it — is skipped, so
/// the result is in general **not** fine.
///
/// The result is TOPCOM's, simplex for simplex. TOPCOM finds the simplex
/// behind a boundary facet by computing the facet's link in the whole
/// triangulation; here each boundary facet carries it, which changes the cost
/// but not the outcome.
///
/// # Errors
///
/// [`Error::NotFullRank`] if `om` has no basis with a non-zero sign (which a
/// [`Chirotope`](crate::om::Chirotope) never lacks, since it is only built for
/// full-rank configurations), and whatever [`OrientedMatroid::sign`] returns.
pub fn placing(om: &impl OrientedMatroid) -> Result<Triangulation> {
    let (n, rank) = (om.n(), om.rank());
    let Some(start) = om.first_basis()? else {
        return Err(Error::NotFullRank {
            rank: 0,
            rows: rank,
        });
    };

    let mut simplices = vec![start.clone()];
    // The boundary of the triangulation so far: each facet in exactly one
    // simplex, mapped to the vertex of that simplex opposite it.
    let mut boundary: FastHashMap<Simplex, usize> = FastHashMap::default();
    toggle_facets(&mut boundary, &start);

    for i in (0..n).filter(|&i| !start.contains(i)) {
        // Every visible facet is decided against the boundary *before* `i`:
        // TOPCOM adds the new simplices as it goes, but a new simplex
        // `F ∪ {i}` only changes the link of `F` itself, so the order does not
        // matter. The boundary is then updated by symmetric difference, which
        // cancels every facet now shared by two simplices.
        let mut new_facets: FastHashMap<Simplex, usize> = FastHashMap::default();
        for (facet, &apex) in &boundary {
            let side_i = om.sign_appended(facet, i)?;
            if side_i.is_zero() {
                continue;
            }
            if om.sign_appended(facet, apex)? == side_i {
                continue;
            }
            let mut s = facet.clone();
            s.insert(i);
            toggle_facets(&mut new_facets, &s);
            simplices.push(s);
        }
        for (facet, apex) in new_facets {
            if boundary.remove(&facet).is_none() {
                boundary.insert(facet, apex);
            }
        }
    }

    Triangulation::new(n, rank, simplices.into_iter().collect())
}

/// Toggle every facet of `simplex` in `set`, recording its opposite vertex: a
/// symmetric difference with the boundary of `simplex`.
fn toggle_facets(set: &mut FastHashMap<Simplex, usize>, simplex: &Simplex) {
    for v in simplex {
        let mut f = simplex.clone();
        f.remove(v);
        if set.remove(&f).is_none() {
            set.insert(f, v);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{PointConfig, VectorConfig};

    #[test]
    fn square_with_an_interior_point_placed_last() {
        let p = PointConfig::from_row_major(&[0, 0, 2, 0, 2, 2, 0, 2, 1, 1], 5, 2).unwrap();
        let t = placing(&p.chirotope().unwrap()).unwrap();
        assert_eq!(t.to_vecs(), vec![vec![0, 1, 2], vec![0, 2, 3]]);
        assert!(!t.is_fine());
        assert_eq!(t.unused(), Simplex::from([4]));
    }

    #[test]
    fn interior_point_placed_early_is_used() {
        let p = PointConfig::from_row_major(&[0, 0, 2, 0, 1, 1, 2, 2, 0, 2], 5, 2).unwrap();
        let chi = p.chirotope().unwrap();
        let t = placing(&chi).unwrap();
        assert!(t.is_fine());
        assert_eq!(t.check(&chi), Ok(()));
    }

    #[test]
    fn rank_one() {
        let v = VectorConfig::from_row_major(&[1, -1, 2, -3], 4, 1).unwrap();
        let t = placing(&v.chirotope().unwrap()).unwrap();
        assert_eq!(t.to_vecs(), vec![vec![0], vec![1]]);
    }
}
