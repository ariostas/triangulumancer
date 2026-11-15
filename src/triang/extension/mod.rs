//! The extension graph: every triangulation, by a backtracking search over
//! partial triangulations.
//!
//! A port of TOPCOM's `SymmetricExtensionGraph` with the trivial group
//! (`lib-src/SymmetricExtensionGraph.cc:945-1507`; `TOPCOM.md` section 7).
//! Unlike the [flip graph](super::flipgraph) it does not use flips, so it
//! finds every triangulation, including those not connected to the others by
//! flips.
//!
//! # The search
//!
//! A node is a *partial triangulation*: a set of pairwise compatible simplices
//! from the universe of the [`Admissibles`] table. Each carries
//!
//! * its *admissibles*, the simplices still compatible with all of it, and
//! * its *free interior facets*: the interior facets
//!   ([`Incidences`]) covered an odd number of times.
//!
//! In a triangulation every interior facet is shared by exactly two
//! simplices, so a partial triangulation is complete exactly when it has no
//! free interior facet. A node is extended by each of its admissibles in
//! canonical order, and a simplex tried as a child is removed from the
//! admissibles of every later sibling's subtree. So every path adds
//! simplices in increasing order and a triangulation `t₁ < t₂ < … < tₖ` has
//! exactly one path to it — each triangulation is produced **exactly once**,
//! with no visited set (`TOPCOM.md` section 7.6(a)).
//!
//! The pruning is TOPCOM's lexicographic rule (section 7.6(b)): the
//! lexicographically smallest free interior facet must be covered by some
//! later simplex, and the facets of a simplex are all at least the one that
//! drops its largest label, which only grows along the candidates. So once
//! that facet of a candidate is beyond the smallest free facet, no later
//! candidate can cover it either and the rest of the siblings are cut. The
//! same test applied to a child's first admissible discards dead children
//! before they are entered.
//!
//! # Memory
//!
//! The search is depth-first over an explicit stack; each level holds its
//! admissibles and free facets as bitsets over the universe and the interior
//! facets. Nothing grows with the output, so [`ExtensionGraph`] yields
//! triangulations one at a time in constant memory (`PLAN.md` section 5.4) —
//! besides the tables, which are built up front and take
//! `|universe|² / 8` bytes.
//!
//! # `only_fine`
//!
//! With `only_fine` the universe is restricted to *empty* simplices
//! (`TOPCOM.md` section 7.7): every triangulation found is then fine, and
//! every fine triangulation is found. This is a pruning of the search, not a
//! filter on its output. When some element lies in no empty simplex — a
//! repeated point, a zero vector — no fine triangulation exists and nothing
//! is yielded, which is also what the flip graph does
//! ([`FlipGraph`](super::FlipGraph)).

mod admissibles;
mod bits;

pub use admissibles::Admissibles;

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, PoisonError};

use super::Triangulation;
use crate::error::{Error, Result};
use crate::om::{Facets, Incidences, OrientedMatroid, circuits};
use crate::simplex::{Simplex, SimplexSet};
use crate::symmetry::{Symmetry, TrivialGroup};

/// A depth-first search of the extension graph, as an iterator over every
/// triangulation.
///
/// The tables are built by the constructor; the search itself is lazy, does
/// no chirotope evaluations and cannot fail, so the items are plain
/// [`Triangulation`]s. The order is deterministic — depth-first, children in
/// canonical order — but it is not canonical; [`all_triangulations`] sorts.
/// [`Iterator::count`] runs the search without building the triangulations,
/// and [`par_count`](Self::par_count) does so on several threads.
///
/// Generic over the [`Symmetry`] group at the point TOPCOM consults its own
/// for output: each complete triangulation is yielded only if it is the
/// representative of its orbit. With [`TrivialGroup`], the only implementer,
/// that is a constant. TOPCOM also prunes *partial* triangulations that are
/// not lexicographically minimal in their orbit (`child_is_lexmin`,
/// `SymmetricExtensionGraphNode.cc:195-294`); a real group would want that
/// as well, which needs a method the trait does not have yet. Uniqueness
/// does not depend on the group: it is carried by the increasing-order rule
/// of the [module documentation](self).
pub struct ExtensionGraph<S = TrivialGroup> {
    symmetry: S,
    tables: Tables,
    search: Search,
}

/// Everything the search reads and never changes.
#[derive(Debug)]
struct Tables {
    n: usize,
    rank: usize,
    only_fine: bool,
    admissibles: Admissibles,
    /// The interior facets of universe simplex `i`, as indices into the
    /// canonically ordered interior facets, are
    /// `facets[facet_start[i]..facet_start[i + 1]]`.
    facets: Vec<u32>,
    facet_start: Vec<usize>,
    /// The number of `u64` words in a bitset over the interior facets.
    facet_words: usize,
    /// For universe simplex `i`, the number of interior facets
    /// lexicographically smaller than its smallest facet: a node whose
    /// smallest free facet has a lower index cannot be completed by
    /// simplex `i` or by any later one.
    first_coverable: Vec<usize>,
}

/// The state of a depth-first search of a subtree. Level `k` of the stack
/// is a node with `k` simplices more than the subtree's root.
#[derive(Debug, Default)]
struct Search {
    /// Level `k`'s admissibles, `admissibles[k * words..(k + 1) * words]`.
    admissibles: Vec<u64>,
    /// Level `k`'s free interior facets, likewise.
    free: Vec<u64>,
    /// The next candidate position to try at each level; one entry per level.
    cursor: Vec<usize>,
    /// The universe positions of the simplices of the deepest node, plus the
    /// last simplex of a triangulation just found.
    chosen: Vec<usize>,
    /// Whether `chosen` ends in a completed triangulation's last simplex.
    found: bool,
}

/// What [`Search::advance`] stopped on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Step {
    /// A triangulation, in `chosen`.
    Found,
    /// The subtree is exhausted.
    Done,
    /// The node budget ran out first; call again to continue.
    Budget,
}

impl ExtensionGraph<TrivialGroup> {
    /// The extension graph of `om` with the trivial group: every
    /// triangulation, or with `only_fine` every fine one.
    ///
    /// # Errors
    ///
    /// [`Error::EmptyConfiguration`] for rank zero, and whatever
    /// [`OrientedMatroid::sign`] returns.
    pub fn new(om: &impl OrientedMatroid, only_fine: bool) -> Result<Self> {
        Self::with_symmetry(om, TrivialGroup, only_fine)
    }
}

impl<S: Symmetry> ExtensionGraph<S> {
    /// As [`new`](ExtensionGraph::new), for a symmetry group.
    ///
    /// Builds every table up front: the circuits, the [`Admissibles`], the
    /// configuration's [`Facets`] and the interior facets of every simplex in
    /// the universe (`TOPCOM.md` section 7.2). That is `C(n, rank + 1)`
    /// circuits, `C(n, rank - 1)` cocircuits, and a table quadratic in the
    /// number of simplices.
    ///
    /// # Errors
    ///
    /// As [`new`](ExtensionGraph::new).
    pub fn with_symmetry(om: &impl OrientedMatroid, symmetry: S, only_fine: bool) -> Result<Self> {
        let tables = Tables::new(om, only_fine)?;
        let search = Search::root(&tables);
        Ok(Self {
            symmetry,
            tables,
            search,
        })
    }

    /// The simplices a triangulation may use: every basis, or every empty one
    /// with `only_fine`, and which pairs of them are compatible.
    pub fn admissibles(&self) -> &Admissibles {
        &self.tables.admissibles
    }

    /// The number of triangulations not yet yielded, counted on `threads`
    /// threads (at least one), without building them.
    ///
    /// [`Iterator::count`] on one thread. The search tree is split between
    /// the threads on demand: a thread that runs out of work takes the
    /// untried siblings nearest the root from one that has some, so the
    /// load stays balanced however lopsided the tree. The memory used is
    /// still independent of the number of triangulations.
    pub fn par_count(self, threads: usize) -> usize
    where
        S: Sync,
    {
        let Self {
            symmetry,
            tables,
            search,
        } = self;
        parallel_count(&symmetry, &tables, search, threads.max(1), SPLIT_INTERVAL)
    }
}

impl Tables {
    fn new(om: &impl OrientedMatroid, only_fine: bool) -> Result<Self> {
        let (n, rank) = (om.n(), om.rank());
        if rank == 0 {
            return Err(Error::EmptyConfiguration);
        }
        let admissibles = Admissibles::from_circuits(om, &circuits(om)?, only_fine)?;
        let config_facets = Facets::new(om)?;
        let incidences =
            Incidences::of_bases(om, &config_facets, admissibles.universe().iter().cloned())?;

        // Number the interior facets in canonical order, so that the
        // smallest free one is the lowest set bit.
        let mut index: BTreeMap<&Simplex, u32> = BTreeMap::new();
        for (_, incidence) in incidences.iter() {
            for f in &incidence.interior {
                index.insert(f, 0);
            }
        }
        let n_facets = index.len();
        let n_facets_u32 =
            u32::try_from(n_facets).map_err(|_| Error::Overflow("too many interior facets"))?;
        for (i, slot) in (0..n_facets_u32).zip(index.values_mut()) {
            *slot = i;
        }
        let sorted: Vec<&Simplex> = index.keys().copied().collect();

        let mut facets = Vec::new();
        let mut facet_start = vec![0];
        let mut first_coverable = Vec::with_capacity(admissibles.len());
        for s in admissibles.universe() {
            let incidence = incidences
                .get(s)
                .ok_or(Error::Internal("a universe simplex is not a basis"))?;
            facets.extend(incidence.interior.iter().map(|f| index[f]));
            facet_start.push(facets.len());
            let mut smallest = s.clone();
            if let Some(last) = s.last() {
                smallest.remove(last);
            }
            first_coverable.push(sorted.partition_point(|f| **f < smallest));
        }

        Ok(Self {
            n,
            rank,
            only_fine,
            admissibles,
            facets,
            facet_start,
            facet_words: n_facets.div_ceil(64),
            first_coverable,
        })
    }

    /// Whether the simplices at `chosen` use every element.
    fn is_fine(&self, chosen: &[usize]) -> bool {
        let universe = self.admissibles.universe();
        let used = chosen
            .iter()
            .fold(Simplex::new(), |acc, &i| &acc | &universe[i]);
        used.len() == self.n
    }

    /// The triangulation of the simplices at `chosen`.
    fn triangulation(&self, chosen: &[usize]) -> Triangulation {
        let universe = self.admissibles.universe();
        // `chosen` is increasing (a child only takes simplices after its
        // parent's) and the universe is in canonical order, so this is sorted.
        let simplices =
            SimplexSet::from_sorted_vec(chosen.iter().map(|&i| universe[i].clone()).collect());
        Triangulation::from_trusted(self.n, self.rank, simplices)
    }
}

impl Search {
    /// The whole search: the root has no simplices, everything admissible
    /// and nothing free. Empty when `only_fine` and some element is in no
    /// empty simplex, since then no triangulation is fine.
    fn root(t: &Tables) -> Self {
        let universe = t.admissibles.universe();
        if t.only_fine {
            let used = universe.iter().fold(Simplex::new(), |acc, s| &acc | s);
            if used.len() != t.n {
                return Self::default();
            }
        }
        let mut root = vec![0; t.admissibles.words()];
        for i in 0..universe.len() {
            bits::set(&mut root, i);
        }
        Self {
            admissibles: root,
            free: vec![0; t.facet_words],
            cursor: vec![0],
            chosen: Vec::new(),
            found: false,
        }
    }

    /// Advance to the next triangulation, leaving its simplices in
    /// `chosen`, visiting at most `budget` nodes (which is decreased).
    fn advance(&mut self, t: &Tables, budget: &mut u64) -> Step {
        let words = t.admissibles.words();
        let fw = t.facet_words;
        if self.found {
            self.chosen.pop();
            self.found = false;
        }
        loop {
            let Some(k) = self.cursor.len().checked_sub(1) else {
                return Step::Done;
            };
            if *budget == 0 {
                return Step::Budget;
            }
            *budget -= 1;
            let level = k * words..(k + 1) * words;
            let free_level = k * fw..(k + 1) * fw;
            let Some(s) = bits::next(&self.admissibles[level.clone()], self.cursor[k]) else {
                self.pop();
                continue;
            };
            self.cursor[k] = s + 1;

            // The lexicographic cut: if the smallest free facet comes before
            // every facet of `s`, nothing from `s` on can cover it.
            let min_free = bits::next(&self.free[free_level.clone()], 0);
            if min_free.is_some_and(|m| m < t.first_coverable[s]) {
                self.pop();
                continue;
            }

            // The child: `s` added. Its free facets flip those of `s`.
            let next_free = (k + 1) * fw..(k + 2) * fw;
            if self.free.len() < next_free.end {
                self.free.resize(next_free.end, 0);
            }
            self.free.copy_within(free_level, next_free.start);
            for &f in &t.facets[t.facet_start[s]..t.facet_start[s + 1]] {
                bits::flip(&mut self.free[next_free.clone()], f as usize);
            }
            let Some(min_free) = bits::next(&self.free[next_free], 0) else {
                // Complete.
                self.chosen.push(s);
                if !t.only_fine || t.is_fine(&self.chosen) {
                    self.found = true;
                    return Step::Found;
                }
                self.chosen.pop();
                continue;
            };

            // Its admissibles: compatible with `s`, and after `s` — every
            // earlier sibling is forbidden, and the parent's admissibles
            // before `s` are exactly those siblings.
            let next_level = (k + 1) * words..(k + 2) * words;
            if self.admissibles.len() < next_level.end {
                self.admissibles.resize(next_level.end, 0);
            }
            let row = t.admissibles.row(s);
            let first_word = s / 64;
            let (parent, child) = self.admissibles.split_at_mut(next_level.start);
            let parent = &parent[level];
            let child = &mut child[..words];
            child[..first_word].fill(0);
            for w in first_word..words {
                child[w] = parent[w] & row[w];
            }
            child[first_word] &= !(u64::MAX >> (63 - s % 64));

            // Dead ends: nothing admissible, or the smallest free facet
            // cannot be covered by any admissible (TOPCOM's
            // `_noncoverable_freefacet_lex`).
            let Some(first) = bits::next(child, first_word) else {
                continue;
            };
            if min_free < t.first_coverable[first] {
                continue;
            }
            self.cursor.push(first);
            self.chosen.push(s);
        }
    }

    /// Leave the deepest level.
    fn pop(&mut self) {
        self.cursor.pop();
        if !self.cursor.is_empty() {
            self.chosen.pop();
        }
    }

    /// Hand over the untried siblings of the shallowest level that has any,
    /// as a search of their own, and give them up here.
    fn split(&mut self, t: &Tables) -> Option<Self> {
        let words = t.admissibles.words();
        let fw = t.facet_words;
        // Leave the deepest level alone: it is about to be finished anyway.
        let levels = self.cursor.len().saturating_sub(1);
        let k = (0..levels).find(|&k| {
            bits::next(
                &self.admissibles[k * words..(k + 1) * words],
                self.cursor[k],
            )
            .is_some()
        })?;
        let other = Self {
            admissibles: self.admissibles[k * words..(k + 1) * words].to_vec(),
            free: self.free[k * fw..(k + 1) * fw].to_vec(),
            cursor: vec![self.cursor[k]],
            // The simplices of the level-`k` node: the first `k` beyond this
            // search's own root.
            chosen: self.chosen[..self.chosen.len() - (levels - k)].to_vec(),
            found: false,
        };
        self.cursor[k] = usize::MAX;
        Some(other)
    }
}

impl<S: Symmetry> Iterator for ExtensionGraph<S> {
    type Item = Triangulation;

    fn next(&mut self) -> Option<Triangulation> {
        let mut unlimited = u64::MAX;
        while self.search.advance(&self.tables, &mut unlimited) == Step::Found {
            let t = self.tables.triangulation(&self.search.chosen);
            // TOPCOM's output predicate; always true for the trivial group.
            if self.symmetry.is_orbit_representative(&t) {
                return Some(t);
            }
        }
        None
    }

    fn count(mut self) -> usize {
        let mut unlimited = u64::MAX;
        count_subtree(
            &self.symmetry,
            &self.tables,
            &mut self.search,
            &mut unlimited,
        )
    }
}

/// Count the triangulations of a search, as far as `budget` allows.
fn count_subtree<S: Symmetry>(
    symmetry: &S,
    t: &Tables,
    search: &mut Search,
    budget: &mut u64,
) -> usize {
    let mut count = 0;
    while search.advance(t, budget) == Step::Found {
        if symmetry.is_trivial()
            || symmetry.is_orbit_representative(&t.triangulation(&search.chosen))
        {
            count += 1;
        }
    }
    count
}

/// The work shared by the threads of [`ExtensionGraph::par_count`].
struct Pool {
    /// Searches nobody has started.
    queue: Vec<Search>,
    /// Threads running a search.
    busy: usize,
    /// Threads waiting for one.
    waiting: usize,
}

/// The node budget between checks for idle threads.
const SPLIT_INTERVAL: u64 = 1 << 12;

fn parallel_count<S: Symmetry + Sync>(
    symmetry: &S,
    t: &Tables,
    search: Search,
    threads: usize,
    interval: u64,
) -> usize {
    let pool = Mutex::new(Pool {
        queue: vec![search],
        busy: 0,
        waiting: 0,
    });
    let wake = Condvar::new();
    let total = AtomicUsize::new(0);
    // A poisoned lock means another worker panicked, which `scope` re-raises;
    // the pool itself is always left consistent.
    let lock = || pool.lock().unwrap_or_else(PoisonError::into_inner);
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                let mut count = 0;
                loop {
                    let mut guard = lock();
                    let mut search = loop {
                        if let Some(s) = guard.queue.pop() {
                            guard.busy += 1;
                            break Some(s);
                        }
                        if guard.busy == 0 {
                            wake.notify_all();
                            break None;
                        }
                        guard.waiting += 1;
                        guard = wake.wait(guard).unwrap_or_else(PoisonError::into_inner);
                        guard.waiting -= 1;
                    };
                    drop(guard);
                    let Some(search) = search.as_mut() else {
                        break;
                    };
                    loop {
                        let mut budget = interval;
                        count += count_subtree(symmetry, t, search, &mut budget);
                        if budget > 0 {
                            break;
                        }
                        let mut guard = lock();
                        if guard.waiting > guard.queue.len() {
                            if let Some(other) = search.split(t) {
                                guard.queue.push(other);
                                wake.notify_one();
                            }
                        }
                    }
                    let mut guard = lock();
                    guard.busy -= 1;
                    if guard.busy == 0 && guard.queue.is_empty() {
                        wake.notify_all();
                    }
                }
                total.fetch_add(count, Ordering::Relaxed);
            });
        }
    });
    total.into_inner()
}

/// Every triangulation of `om`, or with `only_fine` every fine one, in
/// canonical order: TOPCOM's `points2alltriangs` / `points2allfinetriangs`,
/// and 0.3's `all_triangulations`. See [`ExtensionGraph`] and the
/// [module documentation](self).
///
/// # Errors
///
/// [`Error::EmptyConfiguration`] for rank zero, and whatever
/// [`OrientedMatroid::sign`] returns.
pub fn all_triangulations(
    om: &impl OrientedMatroid,
    only_fine: bool,
) -> Result<Vec<Triangulation>> {
    let mut out: Vec<Triangulation> = ExtensionGraph::new(om, only_fine)?.collect();
    out.sort();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{PointConfig, VectorConfig};
    use crate::triang::all_connected;

    fn grid() -> PointConfig {
        let pts: Vec<i64> = (0..3)
            .flat_map(|x| (0..3).flat_map(move |y| [x, y]))
            .collect();
        PointConfig::from_row_major(&pts, 9, 2).unwrap()
    }

    #[test]
    fn the_3x3_grid() {
        let chi = grid().chirotope().unwrap();
        let all = all_triangulations(&chi, false).unwrap();
        assert_eq!(all.len(), 387);
        assert_eq!(all, all_connected(&chi, false).unwrap());
        let fine = all_triangulations(&chi, true).unwrap();
        assert_eq!(fine.len(), 64);
        assert_eq!(fine, all_connected(&chi, true).unwrap());
        for t in all.iter().step_by(20) {
            assert_eq!(t.check(&chi), Ok(()));
        }
        assert_eq!(ExtensionGraph::new(&chi, false).unwrap().count(), 387);
        for threads in [1, 2, 7] {
            assert_eq!(
                ExtensionGraph::new(&chi, false).unwrap().par_count(threads),
                387
            );
            assert_eq!(
                ExtensionGraph::new(&chi, true).unwrap().par_count(threads),
                64
            );
        }
        // Counting after some have been yielded counts the rest.
        let mut g = ExtensionGraph::new(&chi, false).unwrap();
        g.by_ref().take(100).for_each(drop);
        assert_eq!(g.par_count(3), 287);
    }

    #[test]
    fn splitting_the_search_between_threads() {
        // Split as often as possible: after every node.
        let chi = grid().chirotope().unwrap();
        for only_fine in [false, true] {
            let tables = Tables::new(&chi, only_fine).unwrap();
            let expected = if only_fine { 64 } else { 387 };
            for threads in [1, 2, 5, 16] {
                let search = Search::root(&tables);
                let n = parallel_count(&TrivialGroup, &tables, search, threads, 1);
                assert_eq!(n, expected, "{threads} threads");
            }
        }
    }

    #[test]
    fn split_searches_partition_the_triangulations() {
        let chi = grid().chirotope().unwrap();
        let tables = Tables::new(&chi, false).unwrap();
        let mut search = Search::root(&tables);
        let mut all = Vec::new();
        let mut parts = Vec::new();
        let mut budget = 50;
        while search.advance(&tables, &mut budget) == Step::Found {
            all.push(tables.triangulation(&search.chosen));
        }
        parts.push(search.split(&tables).expect("something to split"));
        parts.push(search);
        while let Some(mut part) = parts.pop() {
            let mut budget = 30;
            loop {
                match part.advance(&tables, &mut budget) {
                    Step::Found => all.push(tables.triangulation(&part.chosen)),
                    Step::Done => break,
                    Step::Budget => {
                        if let Some(other) = part.split(&tables) {
                            parts.push(other);
                        }
                        budget = 30;
                    }
                }
            }
        }
        let n = all.len();
        all.sort();
        all.dedup();
        assert_eq!((n, all.len()), (387, 387));
        assert_eq!(all, all_connected(&chi, false).unwrap());
    }

    #[test]
    fn each_triangulation_once() {
        let chi = grid().chirotope().unwrap();
        let mut found: Vec<Triangulation> = ExtensionGraph::new(&chi, false).unwrap().collect();
        let n = found.len();
        found.sort();
        found.dedup();
        assert_eq!(found.len(), n);
    }

    #[test]
    fn a_single_simplex() {
        let p = PointConfig::from_row_major(&[0, 0, 1, 0, 0, 1], 3, 2).unwrap();
        let chi = p.chirotope().unwrap();
        let all = all_triangulations(&chi, false).unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].to_vecs(), vec![vec![0, 1, 2]]);
    }

    #[test]
    fn interior_point() {
        let p = PointConfig::from_row_major(&[0, 0, 3, 0, 0, 3, 1, 1], 4, 2).unwrap();
        let chi = p.chirotope().unwrap();
        let all = all_triangulations(&chi, false).unwrap();
        assert_eq!(all.len(), 2);
        let fine = all_triangulations(&chi, true).unwrap();
        assert_eq!(fine.len(), 1);
        assert!(fine[0].is_fine());
    }

    #[test]
    fn no_fine_triangulation_with_duplicates_or_zero_vectors() {
        let p = PointConfig::from_row_major(&[0, 0, 1, 0, 0, 1, 1, 0], 4, 2).unwrap();
        let chi = p.chirotope().unwrap();
        assert_eq!(all_triangulations(&chi, false).unwrap().len(), 2);
        assert!(all_triangulations(&chi, true).unwrap().is_empty());

        // An interior duplicate: the other points still triangulate.
        let p = PointConfig::from_row_major(&[0, 0, 3, 0, 0, 3, 1, 1, 1, 1], 5, 2).unwrap();
        let chi = p.chirotope().unwrap();
        assert_eq!(all_triangulations(&chi, false).unwrap().len(), 3);
        assert!(all_triangulations(&chi, true).unwrap().is_empty());

        let v = VectorConfig::from_row_major(&[1, 0, 0, 1, 0, 0, 1, 1], 4, 2).unwrap();
        let chi = v.chirotope().unwrap();
        assert_eq!(all_triangulations(&chi, false).unwrap().len(), 2);
        assert!(all_triangulations(&chi, true).unwrap().is_empty());
    }
}
