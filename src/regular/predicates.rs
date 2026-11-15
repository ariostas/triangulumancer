//! The two geometric predicates, and the symbolic perturbation that makes the
//! power test total.
//!
//! Both are determinant signs over [`arith::sign_of_det_of`]:
//!
//! * `orient(p₀, …, p_d) = sign det [pₖ − p₀]ₖ₌₁..d`, a `d × d` determinant of
//!   coordinate differences (`CGAL.md` section 4.1);
//! * `power_side(cell, q) = (−1)^d · sign det [vᵢ − q | ℓᵢ − ℓ_q]ᵢ₌₀..d`, the
//!   same thing one dimension up on lifted points (`CGAL.md` sections 4.2 and
//!   4.3). It is positive when `q̂` lies strictly below the hyperplane through
//!   the lifted vertices of a positively oriented cell, i.e. when `q` is in
//!   conflict with the cell.
//!
//! # Ties
//!
//! A zero from the power test (`d + 2` lifted points on a common hyperplane:
//! four corners of a unit square, under Delaunay) is resolved exactly as CGAL's
//! `perturbed_power_side_of_power_sphere` resolves it
//! (`Regular_triangulation.h`; `CGAL.md` section 4.4). That procedure is a
//! symbolic perturbation of the *lifts*: `ℓⱼ ↦ ℓⱼ + ε^π(j)`, where the point
//! that is lexicographically largest by coordinates gets the most significant
//! (largest) raise. Walking the `d + 2` points from the largest down, the first
//! one whose perturbation term is non-zero decides: the query itself means "not
//! in conflict"; a cell vertex `vᵢ` means "in conflict iff `q` is on the same
//! side of the facet opposite `vᵢ` as `vᵢ`", which is the sign of `q`'s
//! barycentric coordinate on `vᵢ`.
//!
//! Since it is a consistent perturbation of the heights, the result is the
//! unique regular triangulation of the perturbed heights, independent of the
//! insertion order. Using CGAL's order rather than input index (the
//! alternative `CGAL.md` section 4.4 suggests) is what reproduces the 0.3
//! output on every degenerate configuration in the corpus — the unit square,
//! the 3×3 grid, the cubes and cross-polytopes — rather than an equally valid
//! but different triangulation. The order compares the *original* coordinates
//! (not the hull-reduced ones, which CGAL never saw), and breaks the one tie it
//! can have, between duplicate points, by index: the higher index is raised
//! more, so among duplicates with equal lifts the lowest index survives and
//! the rest are hidden (bug B5; CGAL's `std::sort` left that choice
//! unspecified).

use std::cmp::Reverse;

use dashu_int::IBig;

use crate::arith::{self, Sign};
use crate::config::PointConfig;
use crate::error::{Error, Result};

/// The infinite vertex.
pub(crate) const INF: u32 = u32::MAX;

/// The lifts, kept narrow when they allow it so that the power test stays in
/// `i128` arithmetic.
enum Lifts {
    Small(Vec<i64>),
    Big(Vec<IBig>),
}

/// Reduced coordinates, integer lifts, and the perturbation order.
pub(crate) struct Geometry {
    d: usize,
    /// Row-major `n x d`: the hull-reduced coordinates.
    coords: Vec<i64>,
    lifts: Lifts,
    /// The perturbation priority of each point; larger is more significant.
    priority: Vec<u32>,
}

/// Small matrices are built on the stack.
const INLINE: usize = 81;

impl Geometry {
    /// `reduced` is the full-rank, hull-reduced configuration the predicates
    /// run on; `original` supplies the perturbation order.
    pub(crate) fn new(reduced: &PointConfig, original: &PointConfig, lifts: Vec<IBig>) -> Self {
        let n = reduced.len();
        let lifts = match lifts
            .iter()
            .map(|l| i64::try_from(l).ok())
            .collect::<Option<Vec<i64>>>()
        {
            Some(small) => Lifts::Small(small),
            None => Lifts::Big(lifts),
        };
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| original.coords(a).cmp(&original.coords(b)).then(a.cmp(&b)));
        let mut priority = vec![0; n];
        for (rank, &i) in order.iter().enumerate() {
            priority[i] = rank as u32;
        }
        Self {
            d: reduced.dim(),
            coords: reduced.to_row_major(),
            lifts,
            priority,
        }
    }

    /// The dimension.
    pub(crate) fn dim(&self) -> usize {
        self.d
    }

    /// The number of points.
    pub(crate) fn len(&self) -> usize {
        self.priority.len()
    }

    fn point(&self, i: u32) -> &[i64] {
        let i = i as usize;
        &self.coords[i * self.d..(i + 1) * self.d]
    }

    /// The order of the lifts, for the zero-dimensional case.
    pub(crate) fn lowest(&self) -> Option<u32> {
        let n = self.len() as u32;
        (0..n).min_by(|&a, &b| {
            let lift = match &self.lifts {
                Lifts::Small(l) => l[a as usize].cmp(&l[b as usize]),
                Lifts::Big(l) => l[a as usize].cmp(&l[b as usize]),
            };
            lift.then(self.priority(a).cmp(&self.priority(b)))
        })
    }

    fn priority(&self, i: u32) -> u32 {
        self.priority[i as usize]
    }

    /// The orientation of the `d + 1` points `verts`, with `verts[slot]`
    /// replaced by `q` if `slot` is given. No vertex may be [`INF`].
    pub(crate) fn orient(&self, verts: &[u32], subst: Option<(usize, u32)>) -> Sign {
        let d = self.d;
        let at = |k: usize| match subst {
            Some((slot, q)) if slot == k => q,
            _ => verts[k],
        };
        let base = self.point(at(0));
        let fill = |buf: &mut [i128]| {
            for k in 1..=d {
                let p = self.point(at(k));
                for j in 0..d {
                    buf[(k - 1) * d + j] = i128::from(p[j]) - i128::from(base[j]);
                }
            }
        };
        let sign = if d * d <= INLINE {
            let mut buf = [0i128; INLINE];
            fill(&mut buf[..d * d]);
            arith::sign_of_det_of(&buf[..d * d], d)
        } else {
            let mut buf = vec![0i128; d * d];
            fill(&mut buf);
            arith::sign_of_det_of(&buf, d)
        };
        // The shape is right by construction.
        sign.unwrap_or(Sign::Zero)
    }

    /// The unperturbed power test: positive iff `q̂` is strictly below the
    /// hyperplane through the lifted vertices of the positively oriented
    /// finite cell `cell`.
    pub(crate) fn power_side(&self, cell: &[u32], q: u32) -> Sign {
        let d = self.d;
        let n = d + 1;
        let qp = self.point(q);
        let sign = match &self.lifts {
            Lifts::Small(l) => {
                let lq = i128::from(l[q as usize]);
                let fill = |buf: &mut [i128]| {
                    for (i, &v) in cell.iter().enumerate() {
                        let p = self.point(v);
                        for j in 0..d {
                            buf[i * n + j] = i128::from(p[j]) - i128::from(qp[j]);
                        }
                        buf[i * n + d] = i128::from(l[v as usize]) - lq;
                    }
                };
                if n * n <= INLINE {
                    let mut buf = [0i128; INLINE];
                    fill(&mut buf[..n * n]);
                    arith::sign_of_det_of(&buf[..n * n], n)
                } else {
                    let mut buf = vec![0i128; n * n];
                    fill(&mut buf);
                    arith::sign_of_det_of(&buf, n)
                }
            }
            Lifts::Big(l) => {
                let lq = &l[q as usize];
                let mut buf = Vec::with_capacity(n * n);
                for &v in cell {
                    let p = self.point(v);
                    for j in 0..d {
                        buf.push(IBig::from(i128::from(p[j]) - i128::from(qp[j])));
                    }
                    buf.push(&l[v as usize] - lq);
                }
                arith::sign_of_det_of(&buf, n)
            }
        }
        .unwrap_or(Sign::Zero);
        if d % 2 == 0 { sign } else { -sign }
    }

    /// Whether `q` is in conflict with the positively oriented finite cell
    /// `cell`, ties broken by the symbolic perturbation of the module
    /// documentation.
    pub(crate) fn in_conflict(&self, cell: &[u32], q: u32) -> Result<bool> {
        match self.power_side(cell, q) {
            Sign::Positive => return Ok(true),
            Sign::Negative => return Ok(false),
            _ => {}
        }
        let mut order: Vec<(usize, u32)> = cell.iter().copied().enumerate().collect();
        order.push((usize::MAX, q));
        // Most significant raise first. Priorities are distinct per label.
        order.sort_by_key(|&(_, v)| Reverse(self.priority(v)));
        for (slot, _) in order {
            if slot == usize::MAX {
                return Ok(false);
            }
            match self.orient(cell, Some((slot, q))) {
                Sign::Positive => return Ok(true),
                Sign::Negative => return Ok(false),
                _ => {}
            }
        }
        Err(Error::Internal(
            "symbolic perturbation found no deciding term",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry(coords: &[i64], d: usize, lifts: &[i64]) -> Geometry {
        let c = PointConfig::from_row_major(coords, lifts.len(), d).unwrap();
        Geometry::new(&c, &c, lifts.iter().map(|&l| IBig::from(l)).collect())
    }

    #[test]
    fn orientation_and_power_side() {
        // Unit square with Delaunay lifts.
        let g = geometry(&[0, 0, 1, 0, 1, 1, 0, 1], 2, &[0, 1, 2, 1]);
        assert_eq!(g.orient(&[0, 1, 2], None), Sign::Positive);
        assert_eq!(g.orient(&[0, 2, 1], None), Sign::Negative);
        assert_eq!(g.orient(&[0, 1, 2], Some((2, 3))), Sign::Positive);
        // Cocircular: the tie.
        assert_eq!(g.power_side(&[0, 1, 2], 3), Sign::Zero);
        // Point 2 = (1, 1) is lexicographically largest, so it is raised most:
        // the cell {0, 1, 2} gets higher at 3's position iff 3 is on 2's side
        // of the facet {0, 1}, which it is, so 3 is in conflict.
        assert_eq!(g.in_conflict(&[0, 1, 2], 3), Ok(true));
        // Cell {0, 1, 3}: 2 is the query and the largest, so it is raised and
        // not in conflict.
        assert_eq!(g.in_conflict(&[0, 1, 3], 2), Ok(false));

        // A point strictly inside the circumcircle, and one outside, in 1D.
        let g = geometry(&[0, 2, 1, 5], 1, &[0, 4, -1, 25]);
        assert_eq!(g.power_side(&[0, 1], 2), Sign::Positive);
        assert_eq!(g.power_side(&[0, 1], 3), Sign::Negative);
    }

    #[test]
    fn wide_lifts() {
        let big = IBig::ONE << 200;
        let c = PointConfig::from_row_major(&[0, 2, 1], 3, 1).unwrap();
        let g = Geometry::new(&c, &c, vec![big.clone(), big.clone(), big.clone() - 1]);
        assert_eq!(g.power_side(&[0, 1], 2), Sign::Positive);
        let g = Geometry::new(&c, &c, vec![big.clone(), big.clone(), big + 1]);
        assert_eq!(g.power_side(&[0, 1], 2), Sign::Negative);
    }
}
