//! Validity of a triangulation, from chirotope signs alone.
//!
//! A port of TOPCOM's `CheckTriang` (`lib-src/CheckTriang.cc`), without the
//! symmetry bookkeeping. See [`Triangulation::check`] for the criteria.

use std::collections::HashMap;

use super::Triangulation;
use crate::error::{Error, Result};
use crate::om::OrientedMatroid;
use crate::simplex::Simplex;

fn invalid(msg: String) -> Error {
    Error::InvalidTriangulation(msg)
}

pub(super) fn check(om: &impl OrientedMatroid, t: &Triangulation) -> Result<()> {
    let (n, rank) = (om.n(), om.rank());
    if t.n() != n {
        return Err(Error::DimensionMismatch {
            expected: n,
            found: t.n(),
        });
    }
    if t.rank() != rank {
        return Err(Error::DimensionMismatch {
            expected: rank,
            found: t.rank(),
        });
    }
    let Some(reference) = t.iter().next() else {
        return Err(invalid("there are no simplices".into()));
    };

    // (1) Every simplex is a basis; record each facet's apexes on the way.
    let mut facets: HashMap<Simplex, Vec<usize>> = HashMap::new();
    for s in t.iter() {
        if om.sign(s)?.is_zero() {
            return Err(invalid(format!("simplex {s} is degenerate")));
        }
        for v in s {
            let mut f = s.clone();
            f.remove(v);
            facets.entry(f).or_default().push(v);
        }
    }

    // (2) The pseudo-manifold property, with adjacent simplices on opposite
    // sides of their common facet.
    let mut facets: Vec<_> = facets.into_iter().collect();
    facets.sort_unstable(); // report the same violation every time
    for (facet, apexes) in &facets {
        match apexes[..] {
            [i] => {
                let side = om.sign_appended(facet, i)?;
                for j in 0..n {
                    if om.sign_appended(facet, j)? == -side {
                        return Err(invalid(format!(
                            "facet {facet} lies in only one simplex, but {j} is beyond it"
                        )));
                    }
                }
            }
            [i, j] => {
                let (si, sj) = (om.sign_appended(facet, i)?, om.sign_appended(facet, j)?);
                if si != -sj {
                    return Err(invalid(format!(
                        "the simplices through facet {facet} with apexes {i} and {j} overlap"
                    )));
                }
            }
            _ => {
                return Err(invalid(format!(
                    "facet {facet} lies in {} simplices",
                    apexes.len()
                )));
            }
        }
    }

    // (3) A generic point just inside `reference` lies in no other simplex:
    // each other simplex must have some facet with the point strictly on the
    // far side.
    let order = reference.to_vec();
    for s in t.iter().skip(1) {
        let chi = om.sign(s)?;
        // The point is beyond facet `s \ s[k]` exactly when the orientation of
        // (facet, point) is `(-1)^(rank + k) · χ(s)`: replacing `s[k]` by the
        // point flips the orientation of `s`, and moving the point from the
        // end of the ordered basis to position `k` takes `rank - 1 - k`
        // transpositions.
        let mut beyond = if rank % 2 == 0 { chi } else { -chi };
        let mut outside = false;
        for v in s {
            let mut f = s.clone();
            f.remove(v);
            if om.lex_extension_sign(&f, &order)? == beyond {
                outside = true;
                break;
            }
            beyond = -beyond;
        }
        if !outside {
            return Err(invalid(format!("simplices {reference} and {s} overlap")));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::config::{PointConfig, VectorConfig};
    use crate::error::Error;
    use crate::triang::Triangulation;

    fn tri(n: usize, rank: usize, rows: &[&[i64]]) -> Triangulation {
        Triangulation::from_labels(n, rank, &rows.concat(), rank).unwrap()
    }

    fn invalid(t: &Triangulation, chi: &crate::om::Chirotope) -> bool {
        matches!(t.check(chi), Err(Error::InvalidTriangulation(_)))
    }

    #[test]
    fn square() {
        let p = PointConfig::from_row_major(&[0, 0, 1, 0, 1, 1, 0, 1], 4, 2).unwrap();
        let chi = p.chirotope().unwrap();
        assert_eq!(tri(4, 3, &[&[0, 1, 2], &[0, 2, 3]]).check(&chi), Ok(()));
        assert_eq!(tri(4, 3, &[&[0, 1, 3], &[1, 2, 3]]).check(&chi), Ok(()));
        // Half a triangulation, the two triangulations overlaid, crossed
        // diagonals, nothing at all.
        assert!(invalid(&tri(4, 3, &[&[0, 1, 2]]), &chi));
        assert!(invalid(
            &tri(4, 3, &[&[0, 1, 2], &[0, 2, 3], &[0, 1, 3], &[1, 2, 3]]),
            &chi
        ));
        assert!(invalid(&tri(4, 3, &[&[0, 1, 2], &[0, 1, 3]]), &chi));
        assert!(invalid(&tri(4, 3, &[]), &chi));
        // Wrong ground set.
        assert!(matches!(
            Triangulation::from_labels(5, 3, &[0, 1, 2], 3)
                .unwrap()
                .check(&chi),
            Err(Error::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn degenerate_simplices_are_rejected() {
        // 0, 1, 2 collinear.
        let p = PointConfig::from_row_major(&[0, 0, 1, 0, 2, 0, 0, 1], 4, 2).unwrap();
        let chi = p.chirotope().unwrap();
        assert!(invalid(&tri(4, 3, &[&[0, 1, 2]]), &chi));
        assert_eq!(tri(4, 3, &[&[0, 2, 3]]).check(&chi), Ok(()));
        assert_eq!(tri(4, 3, &[&[0, 1, 3], &[1, 2, 3]]).check(&chi), Ok(()));
    }

    #[test]
    fn a_double_cover_is_caught_by_the_covering_test() {
        // Five rays around the origin, and the fan that joins every second
        // one: it winds around twice. Each ray lies in exactly two cones, on
        // opposite sides of it, so only the generic-point test can tell.
        let v = VectorConfig::from_row_major(&[10, 0, 3, 10, -8, 6, -8, -6, 3, -10], 5, 2).unwrap();
        let chi = v.chirotope().unwrap();
        let twice = tri(5, 2, &[&[0, 2], &[2, 4], &[1, 4], &[1, 3], &[0, 3]]);
        match twice.check(&chi) {
            Err(Error::InvalidTriangulation(msg)) => assert!(msg.contains("overlap"), "{msg}"),
            other => panic!("{other:?}"),
        }
        let once = tri(5, 2, &[&[0, 1], &[1, 2], &[2, 3], &[3, 4], &[0, 4]]);
        assert_eq!(once.check(&chi), Ok(()));
    }

    #[test]
    fn vector_configurations() {
        // The whole plane, as a fan of four quadrants.
        let v = VectorConfig::from_row_major(&[1, 0, 0, 1, -1, 0, 0, -1], 4, 2).unwrap();
        let chi = v.chirotope().unwrap();
        let fan = tri(4, 2, &[&[0, 1], &[1, 2], &[2, 3], &[0, 3]]);
        assert_eq!(fan.check(&chi), Ok(()));
        assert!(invalid(&tri(4, 2, &[&[0, 1], &[1, 2], &[2, 3]]), &chi));
        // Rank one: the line, as two rays.
        let v = VectorConfig::from_row_major(&[1, -1, 2], 3, 1).unwrap();
        let chi = v.chirotope().unwrap();
        assert_eq!(tri(3, 1, &[&[0], &[1]]).check(&chi), Ok(()));
        assert_eq!(tri(3, 1, &[&[1], &[2]]).check(&chi), Ok(()));
        assert!(invalid(&tri(3, 1, &[&[0], &[2]]), &chi));
        assert!(invalid(&tri(3, 1, &[&[0], &[1], &[2]]), &chi));
    }
}
