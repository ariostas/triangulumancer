//! The two geometric predicates, and the symbolic perturbation that makes the
//! power test total.
//!
//! Both are determinant signs over [`arith::sign_of_det_of`] (for matrices of
//! size five and up with entries below `2^53`, the filter is first run on
//! `f64` entries built directly from the coordinates, skipping the integer
//! matrix unless it declines):
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

use crate::arith::{self, Sign, filter};
use crate::config::PointConfig;
use crate::error::{Error, Result};

/// The infinite vertex.
pub(crate) const INF: u32 = u32::MAX;

/// The lifts, kept narrow when they allow it so that the power test stays in
/// `i128` arithmetic.
enum Lifts {
    /// Every lift below [`SMALL_LIFT`] in magnitude, so that the difference
    /// of two fits in an `i128`. Scaled `f64` weights usually land here.
    Small(Vec<i128>),
    Big(Vec<IBig>),
}

/// The bound on [`Lifts::Small`].
const SMALL_LIFT: i128 = 1 << 126;

/// Reduced coordinates, integer lifts, and the perturbation order.
pub(crate) struct Geometry {
    d: usize,
    /// Row-major `n x d`: the hull-reduced coordinates.
    coords: Vec<i64>,
    lifts: Lifts,
    /// The perturbation priority of each point; larger is more significant.
    priority: Vec<u32>,
    /// The coordinates, then the lifts, as `f64`, if every one of them is
    /// below `2^52` in magnitude: then every difference of two is an integer
    /// below `2^53`, computed exactly in `f64`, and the filter can be fed
    /// directly (see [`FAST_FROM`]).
    exact: Option<(Vec<f64>, Vec<f64>)>,
}

/// Matrices from this size on try the filter on `f64` entries first when
/// [`Geometry::exact`] allows it. Smaller ones go straight to the exact
/// cofactor expansion of [`arith::sign_of_det_of`], which is cheaper.
const FAST_FROM: usize = 5;

/// Integers below this magnitude, and differences of two of them, are exact
/// in `f64`.
const EXACT_BELOW: i64 = 1 << 52;

/// Matrices are built on the stack up to `12 x 12`: the power test up to
/// dimension 11.
const INLINE: usize = 144;

/// Run `f` on a zeroed scratch matrix of `len` entries: on the stack when it
/// is small (in a buffer sized to match, since zeroing a large one costs as
/// much as a `3 x 3` determinant), on the heap otherwise.
#[inline]
fn with_buffer<R>(len: usize, f: impl FnOnce(&mut [i128]) -> R) -> R {
    if len <= 16 {
        f(&mut [0i128; 16][..len])
    } else if len <= INLINE {
        f(&mut [0i128; INLINE][..len])
    } else {
        f(&mut vec![0i128; len])
    }
}

/// The determinant sign of the `n x n` matrix `m`, skipping the filter if the
/// caller's own attempt on the same matrix has already declined.
fn exact_sign<T: arith::Entry>(m: &[T], n: usize, filtered: bool) -> Sign {
    if filtered {
        arith::sign_of_det_exact_of(m, n)
    } else {
        // The shape is right by construction.
        arith::sign_of_det_of(m, n).unwrap_or(Sign::Zero)
    }
}

impl Geometry {
    /// `reduced` is the full-rank, hull-reduced configuration the predicates
    /// run on; `original` supplies the perturbation order.
    pub(crate) fn new(reduced: &PointConfig, original: &PointConfig, lifts: Vec<IBig>) -> Self {
        let n = reduced.len();
        let lifts = match lifts
            .iter()
            .map(|l| {
                i128::try_from(l)
                    .ok()
                    .filter(|x| x.unsigned_abs() < SMALL_LIFT.unsigned_abs())
            })
            .collect::<Option<Vec<i128>>>()
        {
            Some(small) => Lifts::Small(small),
            None => Lifts::Big(lifts),
        };
        let coords = reduced.to_row_major();
        let exact = match &lifts {
            Lifts::Small(l)
                if coords
                    .iter()
                    .all(|x| x.unsigned_abs() < EXACT_BELOW.unsigned_abs())
                    && l.iter()
                        .all(|x| x.unsigned_abs() < EXACT_BELOW.unsigned_abs().into()) =>
            {
                Some((
                    coords.iter().map(|&x| x as f64).collect(),
                    l.iter().map(|&x| x as f64).collect(),
                ))
            }
            _ => None,
        };
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| original.coords(a).cmp(&original.coords(b)).then(a.cmp(&b)));
        let mut priority = vec![0; n];
        for (rank, &i) in order.iter().enumerate() {
            priority[i] = rank as u32;
        }
        Self {
            d: reduced.dim(),
            coords,
            lifts,
            priority,
            exact,
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
        if d >= FAST_FROM
            && let Some((cf, _)) = &self.exact
        {
            let point = |i: u32| &cf[i as usize * d..(i as usize + 1) * d];
            let base = point(at(0));
            let decided = filter::sign_of_det_integral(d, |buf| {
                for k in 1..=d {
                    let p = point(at(k));
                    for j in 0..d {
                        buf[(k - 1) * d + j] = p[j] - base[j];
                    }
                }
            });
            if let Some(sign) = decided {
                return sign;
            }
        }
        let base = self.point(at(0));
        let fill = |buf: &mut [i128]| {
            for k in 1..=d {
                let p = self.point(at(k));
                for j in 0..d {
                    buf[(k - 1) * d + j] = i128::from(p[j]) - i128::from(base[j]);
                }
            }
        };
        let filtered = d >= FAST_FROM && self.exact.is_some();
        with_buffer(d * d, |buf| {
            fill(buf);
            exact_sign(buf, d, filtered)
        })
    }

    /// The unperturbed power test: positive iff `q̂` is strictly below the
    /// hyperplane through the lifted vertices of the positively oriented
    /// finite cell `cell`.
    pub(crate) fn power_side(&self, cell: &[u32], q: u32) -> Sign {
        let d = self.d;
        let n = d + 1;
        let filtered = n >= FAST_FROM && self.exact.is_some();
        let decided = match &self.exact {
            Some((cf, lf)) if filtered => {
                let point = |i: u32| &cf[i as usize * d..(i as usize + 1) * d];
                let (qp, lq) = (point(q), lf[q as usize]);
                filter::sign_of_det_integral(n, |buf| {
                    for (i, &v) in cell.iter().enumerate() {
                        let p = point(v);
                        for j in 0..d {
                            buf[i * n + j] = p[j] - qp[j];
                        }
                        buf[i * n + d] = lf[v as usize] - lq;
                    }
                })
            }
            _ => None,
        };
        if let Some(sign) = decided {
            return if d % 2 == 0 { sign } else { -sign };
        }
        let qp = self.point(q);
        let sign = match &self.lifts {
            Lifts::Small(l) => {
                let lq = l[q as usize];
                let fill = |buf: &mut [i128]| {
                    for (i, &v) in cell.iter().enumerate() {
                        let p = self.point(v);
                        for j in 0..d {
                            buf[i * n + j] = i128::from(p[j]) - i128::from(qp[j]);
                        }
                        buf[i * n + d] = l[v as usize] - lq;
                    }
                };
                with_buffer(n * n, |buf| {
                    fill(buf);
                    exact_sign(buf, n, filtered)
                })
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
                exact_sign(&buf, n, false)
            }
        };
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

    /// The fast `f64` path must not be taken for `i64::MIN` coordinates:
    /// `i64::MIN.abs()` wraps, and the rounded differences gave a wrong
    /// orientation (found by an audit of the filter).
    #[test]
    fn min_coordinate_does_not_take_fast_path() {
        const MIN: i64 = i64::MIN;
        let (sp, sa, sy, sx) = (
            1_330_303_205_370_881_i64,
            295_058_160_636_927_i64,
            2_133_966_648_241_151_i64,
            1_098_631_412_130_817_i64,
        );
        #[rustfmt::skip]
        let coords = [
            MIN, 0, 0, 0, MIN,
            sp, 0, 0, 0, sy,
            MIN, 1, 0, 0, MIN,
            MIN, 0, 1, 0, MIN,
            MIN, 0, 0, 1, MIN,
            sa, 0, 0, 0, sx,
        ];
        let g = geometry(&coords, 5, &[0; 6]);
        assert!(g.exact.is_none());
        let mut m = Vec::new();
        for k in 1..6 {
            for j in 0..5 {
                m.push(IBig::from(coords[k * 5 + j]) - IBig::from(coords[j]));
            }
        }
        let exact = arith::bareiss::sign_of_det_bigint_of(&m, 5);
        assert_eq!(g.orient(&[0, 1, 2, 3, 4, 5], None), exact);

        // End to end: zero lifts and a constant lift of 2^60 describe the
        // same subdivision, so they must agree.
        let mut c = coords.to_vec();
        c.extend_from_slice(&[MIN, 1, 1, 1, MIN]);
        let cfg = PointConfig::from_row_major(&c, 7, 5).unwrap();
        let zero = crate::regular::from_lifts(&cfg, vec![IBig::ZERO; 7]).unwrap();
        let shifted = crate::regular::from_lifts(&cfg, vec![IBig::ONE << 60; 7]).unwrap();
        assert_eq!(zero, shifted);
    }

    /// A lift of exactly `i128::MIN` must not be classed as small: its `abs`
    /// wraps, and the subtraction of lifts then overflowed.
    #[test]
    fn i128_min_lift() {
        let cfg = PointConfig::from_row_major(&[0, 2, 1], 3, 1).unwrap();
        let min = IBig::from(i128::MIN);
        let a = crate::regular::from_lifts(&cfg, vec![IBig::ZERO, IBig::ZERO, min.clone()]);
        let b = crate::regular::from_lifts(&cfg, vec![IBig::ZERO, IBig::ZERO, min - 1]);
        assert_eq!(a.unwrap(), b.unwrap());
    }
}
