//! The triangulation data structure: an arena of full cells.
//!
//! CGAL's `Triangulation_data_structure` (`CGAL.md` section 3.2), reduced to
//! what insertion needs. The triangulation is of a `d`-sphere: an infinite
//! vertex [`INF`] is joined to every facet of the convex hull, so every cell
//! has exactly `d + 1` neighbours and the hull is not a special case.
//!
//! Cell `c` stores `d + 1` vertices, and for each slot `i` the neighbour
//! across the facet opposite vertex `i` together with the *mirror index*: the
//! slot of `c` in that neighbour. CGAL recomputes mirror indices by a scan and
//! an XOR trick over vertex pointers; here they are stored.
//!
//! Vertices are point labels, so the index of every vertex is known without any
//! search through coordinates (this is what fixes bug B5).

use crate::hash::FastHashMap;
use std::collections::hash_map::Entry;

use super::predicates::INF;
use crate::error::{Error, Result};

/// No cell.
pub(crate) const NONE: u32 = u32::MAX;

/// The arena.
pub(crate) struct Tds {
    /// `d + 1`.
    d1: usize,
    verts: Vec<u32>,
    neigh: Vec<u32>,
    mirror: Vec<u8>,
    alive: Vec<bool>,
    free: Vec<u32>,
    live: usize,
}

impl Tds {
    pub(crate) fn new(d1: usize) -> Self {
        Self {
            d1,
            verts: Vec::new(),
            neigh: Vec::new(),
            mirror: Vec::new(),
            alive: Vec::new(),
            free: Vec::new(),
            live: 0,
        }
    }

    /// `d + 1`, the number of vertices of a cell.
    pub(crate) fn d1(&self) -> usize {
        self.d1
    }

    /// One more than the largest cell id in use, for sizing side tables.
    pub(crate) fn capacity(&self) -> usize {
        self.alive.len()
    }

    /// The number of live cells.
    pub(crate) fn live(&self) -> usize {
        self.live
    }

    /// A new cell with the given vertices and no neighbours yet.
    pub(crate) fn alloc(&mut self, verts: &[u32]) -> u32 {
        let d1 = self.d1;
        debug_assert_eq!(verts.len(), d1);
        self.live += 1;
        if let Some(c) = self.free.pop() {
            let r = c as usize * d1..(c as usize + 1) * d1;
            self.verts[r.clone()].copy_from_slice(verts);
            self.neigh[r.clone()].fill(NONE);
            self.mirror[r].fill(0);
            self.alive[c as usize] = true;
            c
        } else {
            let c = self.alive.len() as u32;
            self.verts.extend_from_slice(verts);
            self.neigh.extend(std::iter::repeat_n(NONE, d1));
            self.mirror.extend(std::iter::repeat_n(0, d1));
            self.alive.push(true);
            c
        }
    }

    /// Delete a cell. Its neighbours' links to it must be overwritten.
    pub(crate) fn release(&mut self, c: u32) {
        self.alive[c as usize] = false;
        self.free.push(c);
        self.live -= 1;
    }

    pub(crate) fn verts(&self, c: u32) -> &[u32] {
        let c = c as usize;
        &self.verts[c * self.d1..(c + 1) * self.d1]
    }

    pub(crate) fn neighbor(&self, c: u32, i: usize) -> u32 {
        self.neigh[c as usize * self.d1 + i]
    }

    pub(crate) fn mirror(&self, c: u32, i: usize) -> usize {
        usize::from(self.mirror[c as usize * self.d1 + i])
    }

    /// Make `a` (across its slot `i`) and `b` (across its slot `j`) neighbours.
    pub(crate) fn link(&mut self, a: u32, i: usize, b: u32, j: usize) {
        let d1 = self.d1;
        self.neigh[a as usize * d1 + i] = b;
        self.mirror[a as usize * d1 + i] = j as u8;
        self.neigh[b as usize * d1 + j] = a;
        self.mirror[b as usize * d1 + j] = i as u8;
    }

    /// The slot of the infinite vertex in `c`, if it has one.
    pub(crate) fn infinite_slot(&self, c: u32) -> Option<usize> {
        self.verts(c).iter().position(|&v| v == INF)
    }

    /// The live cells.
    pub(crate) fn cells(&self) -> impl Iterator<Item = u32> + '_ {
        (0..self.alive.len() as u32).filter(|&c| self.alive[c as usize])
    }

    /// The vertices of the facet of `c` opposite slot `i`, sorted: a key that
    /// identifies the facet from either side.
    pub(crate) fn facet_key(&self, c: u32, i: usize) -> Vec<u32> {
        let mut key: Vec<u32> = self
            .verts(c)
            .iter()
            .enumerate()
            .filter(|&(k, _)| k != i)
            .map(|(_, &v)| v)
            .collect();
        key.sort_unstable();
        key
    }

    /// Link every unlinked facet among `cells` to the one other facet among
    /// them with the same vertices.
    ///
    /// This is the hash-map formulation of hole retriangulation from `CGAL.md`
    /// section 3.6, in place of CGAL's rotor walk. In a valid hole every such
    /// facet has exactly one partner; anything else means the conflict zone
    /// was not a ball, and is reported rather than producing a corrupt mesh.
    pub(crate) fn link_unlinked(&mut self, cells: &[u32]) -> Result<()> {
        let mut open: FastHashMap<Vec<u32>, (u32, usize)> = FastHashMap::default();
        for &c in cells {
            for i in 0..self.d1 {
                if self.neighbor(c, i) != NONE {
                    continue;
                }
                match open.entry(self.facet_key(c, i)) {
                    Entry::Occupied(e) => {
                        let (b, j) = e.remove();
                        self.link(c, i, b, j);
                    }
                    Entry::Vacant(e) => {
                        e.insert((c, i));
                    }
                }
            }
        }
        if open.is_empty() {
            Ok(())
        } else {
            Err(Error::Internal(
                "regular triangulation: a facet of the retriangulated hole has no partner",
            ))
        }
    }

    /// The combinatorial invariants: every link is live and symmetric with a
    /// correct mirror index, neighbours share exactly the facet between them,
    /// and no cell repeats a vertex.
    pub(crate) fn check(&self) -> Result<()> {
        let fail = |what| Err(Error::Internal(what));
        for c in self.cells() {
            let verts = self.verts(c);
            for (k, v) in verts.iter().enumerate() {
                if verts[..k].contains(v) {
                    return fail("TDS check: a cell repeats a vertex");
                }
            }
            for i in 0..self.d1 {
                let n = self.neighbor(c, i);
                if n == NONE || !self.alive.get(n as usize).copied().unwrap_or(false) {
                    return fail("TDS check: a neighbour link is dangling");
                }
                let j = self.mirror(c, i);
                if self.neighbor(n, j) != c || self.mirror(n, j) != i {
                    return fail("TDS check: a neighbour link is not symmetric");
                }
                if self.facet_key(c, i) != self.facet_key(n, j) {
                    return fail("TDS check: neighbours do not share their facet");
                }
            }
        }
        Ok(())
    }
}
