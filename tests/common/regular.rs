//! Independent oracles for regular triangulations (phase 6).
//!
//! These share no code with `triangulumancer::regular` beyond the exact
//! determinant sign, which has its own differential harness
//! (`tests/sign_of_det.rs`):
//!
//! * [`assert_regular`] — the ground-truth certificate of `CGAL.md` section 8
//!   (6): the result is a triangulation (`Triangulation::check` on the
//!   chirotope of the hull-reduced configuration), and every lifted point,
//!   hidden ones included, is on or above the hyperplane through every lifted
//!   simplex. Together these say the triangulation refines the regular
//!   subdivision, whatever tie-breaking was used.
//! * [`brute_force`] — the tie-breaking too: perturb the lifts *numerically*
//!   by the rule the implementation applies symbolically, which makes them
//!   generic, and keep every simplex whose lifted hyperplane has every other
//!   lifted point strictly above it. For generic lifts that set is the unique
//!   regular triangulation.

use dashu_int::IBig;
use triangulumancer::arith::{self, Sign};
use triangulumancer::config::PointConfig;
use triangulumancer::triang::Triangulation;

/// `M` and `F` of the lower-hull test for simplex `sigma` and point `j`:
/// `M = det [x_i 1]` over the simplex and `F = det [x_i l_i 1]` over the
/// simplex then `j`. Expanding `F` along its lift column, `F = -M (l_j - h)`
/// where `h` is the height of the simplex's hyperplane at `x_j`; so `j` is
/// above iff `F * M < 0`, and on it iff `F = 0`.
fn lower_hull_test(
    reduced: &PointConfig,
    lifts: &[IBig],
    sigma: &[usize],
    j: usize,
) -> (Sign, Sign) {
    let d = reduced.dim();
    let mut m: Vec<IBig> = Vec::new();
    for &i in sigma {
        m.extend(reduced.coords(i).unwrap().iter().map(|&x| IBig::from(x)));
        m.push(IBig::ONE);
    }
    let orient = arith::sign_of_det_of(&m, d + 1).unwrap();
    let mut f: Vec<IBig> = Vec::new();
    for &i in sigma.iter().chain([j].iter()) {
        f.extend(reduced.coords(i).unwrap().iter().map(|&x| IBig::from(x)));
        f.push(lifts[i].clone());
        f.push(IBig::ONE);
    }
    (orient, arith::sign_of_det_of(&f, d + 2).unwrap())
}

/// Panic unless `t` is a regular triangulation of `config` for `lifts`, in the
/// weak sense of the module documentation.
pub fn assert_regular(config: &PointConfig, lifts: &[IBig], t: &Triangulation, what: &str) {
    let reduced = config.reduce_to_hull().unwrap();
    assert_eq!(t.rank(), reduced.rows(), "{what}: rank");
    t.check(&reduced.chirotope().unwrap())
        .unwrap_or_else(|e| panic!("{what}: not a triangulation: {e}"));
    for s in t.iter() {
        let sigma = s.to_vec();
        for j in 0..config.len() {
            if s.contains(j) {
                continue;
            }
            let (m, f) = lower_hull_test(&reduced, lifts, &sigma, j);
            assert_ne!(m, Sign::Zero, "{what}: flat simplex {sigma:?}");
            assert_ne!(
                m * f,
                Sign::Positive,
                "{what}: point {j} is below the lifted simplex {sigma:?}"
            );
        }
    }
}

/// The perturbation priority: the rank of each point by original coordinates,
/// then label. A higher priority is a larger (more significant) raise.
pub fn priority(config: &PointConfig) -> Vec<usize> {
    let mut order: Vec<usize> = (0..config.len()).collect();
    order.sort_by_key(|&i| (config.coords(i).unwrap().to_vec(), i));
    let mut p = vec![0; order.len()];
    for (rank, i) in order.into_iter().enumerate() {
        p[i] = rank;
    }
    p
}

/// The regular triangulation by brute force over every simplex, with the
/// lifts perturbed numerically: `l'_j = l_j B^n + B^priority(j)`, `B = 2^bits`.
/// `bits` must exceed the bit size of every orientation cofactor by a margin;
/// for small coordinates and dimensions, 64 is plenty.
pub fn brute_force(config: &PointConfig, lifts: &[IBig], bits: usize) -> Vec<Vec<usize>> {
    let n = config.len();
    let reduced = config.reduce_to_hull().unwrap();
    let r = reduced.rows();
    let prio = priority(config);
    let perturbed: Vec<IBig> = (0..n)
        .map(|j| (&lifts[j] << (bits * n)) + (IBig::ONE << (bits * prio[j])))
        .collect();
    let mut out = Vec::new();
    for s in triangulumancer::simplex::combinations(n, r) {
        let sigma: Vec<usize> = s.to_vec();
        let mut keep = true;
        for j in 0..n {
            if s.contains(j) {
                continue;
            }
            let (m, f) = lower_hull_test(&reduced, &perturbed, &sigma, j);
            if m == Sign::Zero || m * f != Sign::Negative {
                keep = false;
                break;
            }
        }
        if keep {
            out.push(sigma);
        }
    }
    if r == 1 {
        // Zero-dimensional: every "simplex" is a point and nothing is compared.
        let winner = (0..n).min_by_key(|&j| perturbed[j].clone()).unwrap();
        return vec![vec![winner]];
    }
    out.sort();
    out
}
