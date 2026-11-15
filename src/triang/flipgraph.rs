//! The flip graph: every triangulation reachable from a seed by bistellar
//! flips.
//!
//! A port of TOPCOM's `SymmetricFlipGraph` (`lib-src/SymmetricFlipGraph.cc`;
//! `TOPCOM.md` section 6) with two simplifications:
//!
//! * **One visited set.** TOPCOM keeps only two BFS layers and marks the flip
//!   back to the parent so as never to revisit the layer it discarded. Keeping
//!   every triangulation seen in one set makes the marking unnecessary, at the
//!   cost of memory proportional to the output (`TOPCOM.md` section 6.2).
//! * **Naive flip sets.** Each triangulation's flips are recomputed from
//!   scratch ([`flip_reps`]) rather than updated from its parent's
//!   (`TOPCOM.md` section 5.3); the incremental update is a phase 7
//!   optimisation, with this as its oracle.
//!
//! The enumeration is **streaming** (`PLAN.md` section 5.4): [`FlipGraph`] is
//! an iterator that yields each triangulation as it is discovered, so a caller
//! that counts or filters never holds the output. The visited set still grows
//! with it; that is inherent to a graph search without a canonical spanning
//! tree.
//!
//! # `only_fine`
//!
//! With `only_fine`, the search starts from the [`fine`] triangulation and
//! never takes a flip that removes a vertex, so it stays among fine
//! triangulations. The result is the connected component of the fine seed in
//! the subgraph of the flip graph *induced on fine triangulations*: a fine
//! triangulation reachable from it only through non-fine ones is not found.
//! That is TOPCOM's `points2finetriangs` (`TOPCOM.md` section 6.3, and
//! `PLAN.md` section 12, question 2).
//!
//! TOPCOM's fine triangulation is not fine when there are duplicate points or
//! zero vectors, and then no fine triangulation exists at all; the search
//! yields nothing. 0.3 seeded this search with the placing triangulation
//! instead, which is bug B1 (`PLAN.md` section 3).

use std::collections::{HashSet, VecDeque};

use super::flip::flip_reps;
use super::{Triangulation, fine, placing};
use crate::error::Result;
use crate::om::OrientedMatroid;
use crate::symmetry::{Symmetry, TrivialGroup};

/// A breadth-first search of the flip graph, as an iterator over the
/// triangulations it finds, the seed first.
///
/// The order is deterministic: breadth-first, each triangulation's
/// neighbours in the order of their flips ([`flip_reps`]). Once an item is an
/// error the search stops.
///
/// Generic over the [`Symmetry`] group at the points TOPCOM consults its own:
/// the stabiliser of each triangulation where its flips are computed (TOPCOM
/// adds the flips equivalent under it), and the orbit-representative test on
/// output. With [`TrivialGroup`], the only implementer, both are constants. A
/// real group would also have to replace the visited set by one of orbits;
/// that is left to whoever adds one.
pub struct FlipGraph<O, S = TrivialGroup> {
    om: O,
    symmetry: S,
    only_fine: bool,
    visited: HashSet<Triangulation>,
    /// Triangulations found whose flips have not been explored yet.
    frontier: VecDeque<Triangulation>,
    /// Triangulations found and not yet yielded.
    pending: VecDeque<Triangulation>,
    failed: bool,
}

impl<O: OrientedMatroid> FlipGraph<O, TrivialGroup> {
    /// The flip graph of `om` with the trivial group, seeded as 0.3 seeded it
    /// apart from bug B1: with the [`placing`] triangulation, or with the
    /// [`fine`] one when `only_fine`.
    ///
    /// # Errors
    ///
    /// Whatever [`placing`] and [`fine`] return.
    pub fn new(om: O, only_fine: bool) -> Result<Self> {
        Self::with_symmetry(om, TrivialGroup, only_fine)
    }
}

impl<O: OrientedMatroid, S: Symmetry> FlipGraph<O, S> {
    /// As [`new`](FlipGraph::new), for a symmetry group.
    ///
    /// # Errors
    ///
    /// Whatever [`placing`] and [`fine`] return.
    pub fn with_symmetry(om: O, symmetry: S, only_fine: bool) -> Result<Self> {
        let seed = if only_fine { fine(&om)? } else { placing(&om)? };
        Ok(Self::from_seed(om, symmetry, seed, only_fine))
    }

    /// The flip graph of `om` from a given seed, which should be a
    /// triangulation of `om` (it is not checked).
    ///
    /// With `only_fine`, a seed that is not fine yields nothing; see the
    /// [module documentation](self).
    pub fn from_seed(om: O, symmetry: S, seed: Triangulation, only_fine: bool) -> Self {
        let mut graph = Self {
            om,
            symmetry,
            only_fine,
            visited: HashSet::new(),
            frontier: VecDeque::new(),
            pending: VecDeque::new(),
            failed: false,
        };
        if !only_fine || seed.is_fine() {
            graph.discover(seed);
        }
        graph
    }

    /// The number of triangulations found so far, yielded or not.
    pub fn found(&self) -> usize {
        self.visited.len()
    }

    fn discover(&mut self, t: Triangulation) {
        if self.visited.insert(t.clone()) {
            // TOPCOM's output predicate; always true for the trivial group.
            if self.symmetry.is_orbit_representative(&t) {
                self.pending.push_back(t.clone());
            }
            self.frontier.push_back(t);
        }
    }

    /// Explore the flips of the next triangulation on the frontier. Returns
    /// whether there was one.
    fn expand(&mut self) -> Result<bool> {
        let Some(t) = self.frontier.pop_front() else {
            return Ok(false);
        };
        // TOPCOM computes the stabiliser here and adds every flip equivalent
        // under it (`TriangFlips.cc:84-95`); the trivial group's is `()`.
        let _stabilizer = self.symmetry.stabilizer(&t);
        for rep in flip_reps(&self.om, &t, self.only_fine)? {
            let next = rep.expand(&t).apply(&t)?;
            debug_assert!(!self.only_fine || next.is_fine());
            self.discover(next);
        }
        Ok(true)
    }
}

impl<O: OrientedMatroid, S: Symmetry> Iterator for FlipGraph<O, S> {
    type Item = Result<Triangulation>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.failed {
            return None;
        }
        loop {
            if let Some(t) = self.pending.pop_front() {
                return Some(Ok(t));
            }
            match self.expand() {
                Ok(true) => {}
                Ok(false) => return None,
                Err(e) => {
                    self.failed = true;
                    return Some(Err(e));
                }
            }
        }
    }
}

/// Every triangulation in the flip-graph component of the seed, in canonical
/// order: TOPCOM's `points2triangs`, or `points2finetriangs` when `only_fine`,
/// and 0.3's `all_connected_triangulations` with bug B1 fixed. See
/// [`FlipGraph`] and the [module documentation](self).
///
/// # Errors
///
/// Whatever [`placing`], [`fine`] and [`OrientedMatroid::sign`] return.
pub fn all_connected(om: &impl OrientedMatroid, only_fine: bool) -> Result<Vec<Triangulation>> {
    let mut out = FlipGraph::new(om, only_fine)?.collect::<Result<Vec<_>>>()?;
    out.sort();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::PointConfig;

    fn grid() -> PointConfig {
        let pts: Vec<i64> = (0..3)
            .flat_map(|x| (0..3).flat_map(move |y| [x, y]))
            .collect();
        PointConfig::from_row_major(&pts, 9, 2).unwrap()
    }

    #[test]
    fn the_3x3_grid() {
        let chi = grid().chirotope().unwrap();
        let all = all_connected(&chi, false).unwrap();
        assert_eq!(all.len(), 387);
        let fine_only = all_connected(&chi, true).unwrap();
        assert_eq!(fine_only.len(), 64);
        assert!(fine_only.iter().all(Triangulation::is_fine));
        assert_eq!(
            all.iter().filter(|t| t.is_fine()).count(),
            64,
            "every fine triangulation of the grid is reached among fine ones"
        );
        for t in fine_only.iter().take(5) {
            assert_eq!(t.check(&chi), Ok(()));
        }
    }

    #[test]
    fn streaming_yields_the_seed_first_and_is_lazy() {
        let chi = grid().chirotope().unwrap();
        let mut g = FlipGraph::new(&chi, false).unwrap();
        assert_eq!(g.found(), 1);
        assert_eq!(g.next(), Some(Ok(placing(&chi).unwrap())));
        assert_eq!(g.found(), 1, "nothing explored before it is asked for");
        assert!(g.next().is_some());
        assert!(g.found() < 387);
        assert_eq!(g.count() + 2, 387);
    }

    #[test]
    fn duplicates_have_no_fine_triangulations() {
        let p = PointConfig::from_row_major(&[0, 0, 1, 0, 0, 1, 1, 0], 4, 2).unwrap();
        let chi = p.chirotope().unwrap();
        assert!(all_connected(&chi, true).unwrap().is_empty());
        assert_eq!(all_connected(&chi, false).unwrap().len(), 2);
    }

    #[test]
    fn a_non_fine_seed_yields_nothing_when_only_fine() {
        let p = PointConfig::from_row_major(&[0, 0, 3, 0, 0, 3, 1, 1], 4, 2).unwrap();
        let chi = p.chirotope().unwrap();
        let seed = placing(&chi).unwrap();
        assert!(!seed.is_fine());
        assert_eq!(
            FlipGraph::from_seed(&chi, TrivialGroup, seed, true).count(),
            0
        );
        assert_eq!(all_connected(&chi, true).unwrap().len(), 1);
    }
}
