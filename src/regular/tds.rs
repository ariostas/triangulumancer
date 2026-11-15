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
    /// Scratch for [`Tds::link_unlinked`], kept to reuse its allocation.
    open: Vec<(u64, u32, u8)>,
    /// Per-point marks for [`Tds::same_facet`]: `stamp[v] == epoch` iff `v`
    /// is in the facet being compared against.
    stamp: Vec<u32>,
    epoch: u32,
}

/// A well-mixed 64-bit image of a vertex (the SplitMix64 finaliser), summed
/// over a facet's vertices to give a hash independent of their order.
#[inline]
fn mix(v: u32) -> u64 {
    let mut z = u64::from(v).wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
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
            open: Vec::new(),
            stamp: Vec::new(),
            epoch: 0,
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
    /// This is the hashing formulation of hole retriangulation from `CGAL.md`
    /// section 3.6, in place of CGAL's rotor walk. In a valid hole every such
    /// facet has exactly one partner; anything else means the conflict zone
    /// was not a ball, and is reported rather than producing a corrupt mesh.
    ///
    /// Facets are hashed by the sum of [`mix`] over their vertices — the
    /// cell's sum minus the opposite vertex, so `O(d)` per cell — sorted by
    /// hash, and matched by comparing vertex sets within each run of equal
    /// hashes, so a hash collision costs time, never correctness.
    pub(crate) fn link_unlinked(&mut self, cells: &[u32]) -> Result<()> {
        let mut open = std::mem::take(&mut self.open);
        open.clear();
        for &c in cells {
            let verts = self.verts(c);
            let all = verts.iter().fold(0u64, |h, &v| h.wrapping_add(mix(v)));
            for (i, &v) in verts.iter().enumerate() {
                if self.neighbor(c, i) == NONE {
                    open.push((all.wrapping_sub(mix(v)), c, i as u8));
                }
            }
        }
        // Deterministic: the entries are distinct.
        open.sort_unstable();
        let result = self.pair(&open);
        self.open = open;
        result
    }

    /// Match the facets of `open`, sorted by hash; see [`Tds::link_unlinked`].
    fn pair(&mut self, open: &[(u64, u32, u8)]) -> Result<()> {
        let unpaired = Err(Error::Internal(
            "regular triangulation: a facet of the retriangulated hole has no partner",
        ));
        let mut start = 0;
        while start < open.len() {
            let hash = open[start].0;
            // Runs are almost always pairs: scan, rather than search.
            let mut end = start + 1;
            while end < open.len() && open[end].0 == hash {
                end += 1;
            }
            let run = &open[start..end];
            for (k, &(_, a, i)) in run.iter().enumerate() {
                let i = usize::from(i);
                if self.neighbor(a, i) != NONE {
                    continue;
                }
                let mut partner = None;
                for &(_, b, j) in &run[k + 1..] {
                    let j = usize::from(j);
                    if self.neighbor(b, j) == NONE && self.same_facet(a, i, b, j) {
                        partner = Some((b, j));
                        break;
                    }
                }
                match partner {
                    Some((b, j)) => self.link(a, i, b, j),
                    None => return unpaired,
                }
            }
            start = end;
        }
        Ok(())
    }

    /// Whether the facet of `a` opposite slot `i` and that of `b` opposite
    /// slot `j` have the same vertices: mark one, look up the other. Cells
    /// never repeat a vertex and the facets have the same size, so
    /// containment one way suffices.
    fn same_facet(&mut self, a: u32, i: usize, b: u32, j: usize) -> bool {
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.stamp.fill(0);
            self.epoch = 1;
        }
        let (d1, epoch) = (self.d1, self.epoch);
        let mut infinite = false;
        for k in 0..d1 {
            let v = self.verts[a as usize * d1 + k];
            if k == i {
                continue;
            }
            if v == INF {
                infinite = true;
            } else {
                let v = v as usize;
                if v >= self.stamp.len() {
                    self.stamp.resize(v + 1, 0);
                }
                self.stamp[v] = epoch;
            }
        }
        (0..d1).all(|l| {
            let w = self.verts[b as usize * d1 + l];
            l == j
                || if w == INF {
                    infinite
                } else {
                    self.stamp.get(w as usize) == Some(&epoch)
                }
        })
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
