//! Incremental insertion: walk, conflict zone, hole retriangulation.
//!
//! The algorithm of CGAL's `Regular_triangulation` (`CGAL.md` sections 3.4 to
//! 3.6), in full dimension only — the configuration has already been reduced
//! to its affine hull, so there is no dimension growth. Inserting a point `p`:
//!
//! 1. **Locate** it by a remembering visibility walk from the last cell
//!    created, ending in a finite cell whose closure contains `p`, or in the
//!    infinite cell beyond the hull facet `p` was seen through. Regular
//!    triangulations are acyclic for the in-front relation (Edelsbrunner), so
//!    with exact predicates the walk never revisits a cell; it is capped
//!    anyway, and exceeding the cap is an error rather than a hang.
//! 2. If `p` is **not in conflict** with that cell, `p̂` lies above the lower
//!    hull: `p` is hidden and nothing changes.
//! 3. Otherwise gather the **conflict zone** by breadth-first search; it is
//!    connected and star-shaped from `p`.
//! 4. **Retriangulate the hole** by coning each boundary facet to `p`, linking
//!    the new cells outwards through the stored mirror indices and among
//!    themselves by matching facets by hash ([`Tds::link_unlinked`]).
//! 5. A vertex of the zone that is in no new cell has just become **hidden**.
//!
//! The conflict test for an infinite cell is the orientation of its hull
//! facet with `p` in place of the infinite vertex, and on a tie (`p` in the
//! facet's hyperplane) the conflict test of the finite cell across the facet,
//! as CGAL does.
//!
//! In debug builds the full set of invariants — combinatorial, orientation of
//! every cell — is checked after every insertion.

use super::predicates::{Geometry, INF};
use super::tds::{NONE, Tds};
use crate::arith::Sign;
use crate::error::{Error, Result};

/// A triangulation under construction.
pub(crate) struct Builder<'g> {
    g: &'g Geometry,
    tds: Tds,
    /// Where the next walk starts.
    hint: u32,
    /// xorshift state, for the walk's starting facet.
    rng: u64,
    /// Points that are not vertices, in the order they became hidden.
    hidden: Vec<u32>,
    /// Per-cell marks for the conflict search: `2e` in the zone and `2e + 1`
    /// examined and outside it, for the insertion with epoch `e`.
    mark: Vec<u64>,
    epoch: u64,
    /// Per-point marks for finding newly hidden vertices.
    seen: Vec<u64>,
    /// Scratch for [`Builder::insert`]: the conflict zone and the new cells,
    /// kept to reuse their allocations.
    zone: Vec<u32>,
    created: Vec<u32>,
}

impl<'g> Builder<'g> {
    /// The triangulation of one simplex: the `d + 1` affinely independent
    /// points `initial`, and the `d + 1` infinite cells around it.
    pub(crate) fn new(g: &'g Geometry, initial: &[u32]) -> Result<Self> {
        let d1 = g.dim() + 1;
        if initial.len() != d1 {
            return Err(Error::Internal("regular triangulation: initial simplex"));
        }
        let mut s = initial.to_vec();
        match g.orient(&s, None) {
            Sign::Positive => {}
            Sign::Negative => s.swap(0, 1),
            _ => {
                return Err(Error::Internal(
                    "regular triangulation: flat initial simplex",
                ));
            }
        }
        let mut tds = Tds::new(d1);
        let mut cells = vec![tds.alloc(&s)];
        for i in 0..d1 {
            // Replacing the infinite vertex by the vertex across the hull facet
            // must give a negatively oriented cell, hence the swap.
            let mut inf = s.clone();
            inf[i] = INF;
            inf.swap(i, (i + 1) % d1);
            cells.push(tds.alloc(&inf));
        }
        tds.link_unlinked(&cells)?;
        let b = Self {
            g,
            tds,
            hint: cells[0],
            rng: 0x2545_f491_4f6c_dd1d,
            hidden: Vec::new(),
            mark: Vec::new(),
            epoch: 0,
            seen: vec![0; g.len()],
            zone: Vec::new(),
            created: Vec::new(),
        };
        b.debug_check()?;
        Ok(b)
    }

    fn next_random(&mut self, below: usize) -> usize {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        (self.rng % below as u64) as usize
    }

    /// The finite cell across the hull facet of an infinite cell, or the cell
    /// itself if it is finite.
    fn finite_side(&self, c: u32) -> u32 {
        match self.tds.infinite_slot(c) {
            Some(i) => self.tds.neighbor(c, i),
            None => c,
        }
    }

    /// Step 1: a finite cell containing `p`, or an infinite cell `p` sees.
    fn locate(&mut self, p: u32) -> Result<u32> {
        let d1 = self.tds.d1();
        let mut c = self.finite_side(self.hint);
        let mut previous = NONE;
        // Each cell is visited at most once (see the module documentation).
        for _ in 0..=self.tds.live() {
            let start = self.next_random(d1);
            let mut moved = false;
            for k in 0..d1 {
                let i = (start + k) % d1;
                let n = self.tds.neighbor(c, i);
                if n == previous {
                    continue;
                }
                if self.g.orient(self.tds.verts(c), Some((i, p))) == Sign::Negative {
                    previous = c;
                    c = n;
                    moved = true;
                    break;
                }
            }
            if !moved || self.tds.infinite_slot(c).is_some() {
                return Ok(c);
            }
        }
        Err(Error::Internal(
            "regular triangulation: point location did not terminate",
        ))
    }

    /// Whether `p` is in conflict with cell `c`.
    fn in_conflict(&self, c: u32, p: u32) -> Result<bool> {
        let verts = self.tds.verts(c);
        match self.tds.infinite_slot(c) {
            None => self.g.in_conflict(verts, p),
            Some(i) => {
                // The orientation predicate never sees the infinite vertex:
                // substituting p for it gives d + 1 finite points.
                match self.g.orient(verts, Some((i, p))) {
                    Sign::Positive => Ok(true),
                    Sign::Negative => Ok(false),
                    _ => self
                        .g
                        .in_conflict(self.tds.verts(self.tds.neighbor(c, i)), p),
                }
            }
        }
    }

    /// Insert point `p`, or record it as hidden.
    pub(crate) fn insert(&mut self, p: u32) -> Result<()> {
        let start = self.locate(p)?;
        if !self.in_conflict(start, p)? {
            self.hidden.push(p);
            return Ok(());
        }

        // Step 3: the conflict zone.
        self.epoch += 1;
        let (inside, outside) = (2 * self.epoch, 2 * self.epoch + 1);
        self.mark.resize(self.tds.capacity(), 0);
        self.mark[start as usize] = inside;
        let mut zone = std::mem::take(&mut self.zone);
        zone.clear();
        zone.push(start);
        let mut head = 0;
        while head < zone.len() {
            let c = zone[head];
            head += 1;
            for i in 0..self.tds.d1() {
                let n = self.tds.neighbor(c, i);
                let m = self.mark[n as usize];
                if m == inside || m == outside {
                    continue;
                }
                if self.in_conflict(n, p)? {
                    self.mark[n as usize] = inside;
                    zone.push(n);
                } else {
                    self.mark[n as usize] = outside;
                }
            }
        }

        // Step 4: cone every boundary facet to p.
        let mut created = std::mem::take(&mut self.created);
        created.clear();
        let mut verts = Vec::with_capacity(self.tds.d1());
        for &c in &zone {
            for i in 0..self.tds.d1() {
                let n = self.tds.neighbor(c, i);
                if self.mark[n as usize] == inside {
                    continue;
                }
                let j = self.tds.mirror(c, i);
                verts.clear();
                verts.extend_from_slice(self.tds.verts(c));
                verts[i] = p;
                let new = self.tds.alloc(&verts);
                self.tds.link(new, i, n, j);
                created.push(new);
            }
        }
        if created.is_empty() {
            return Err(Error::Internal(
                "regular triangulation: the conflict zone has no boundary",
            ));
        }
        self.tds.link_unlinked(&created)?;

        // Step 5: vertices of the zone that no new cell uses are now hidden.
        let tag = self.epoch;
        for &c in &created {
            for &v in self.tds.verts(c) {
                if v != INF {
                    self.seen[v as usize] = tag;
                }
            }
        }
        for &c in &zone {
            for k in 0..self.tds.d1() {
                let v = self.tds.verts(c)[k];
                if v == INF {
                    continue;
                }
                if self.seen[v as usize] != tag {
                    self.seen[v as usize] = tag;
                    self.hidden.push(v);
                }
            }
        }
        for &c in &zone {
            self.tds.release(c);
        }
        self.hint = created[0];
        self.zone = zone;
        self.created = created;
        self.debug_check()
    }

    /// The invariants of `CGAL.md` section 8 (1) and (2): combinatorial
    /// consistency, every finite cell positively oriented, and every infinite
    /// cell negatively oriented once the vertex across its hull facet replaces
    /// the infinite vertex. The empty-sphere property is the job of the tests'
    /// independent regularity oracle.
    pub(crate) fn check(&self) -> Result<()> {
        self.tds.check()?;
        let fail = |what| Err(Error::Internal(what));
        let mut infinite = 0;
        for c in self.tds.cells() {
            let verts = self.tds.verts(c);
            match self.tds.infinite_slot(c) {
                None => {
                    if self.g.orient(verts, None) != Sign::Positive {
                        return fail("regular triangulation: a finite cell is not positive");
                    }
                }
                Some(i) => {
                    infinite += 1;
                    let n = self.tds.neighbor(c, i);
                    let across = self.tds.verts(n)[self.tds.mirror(c, i)];
                    if across == INF || self.g.orient(verts, Some((i, across))) != Sign::Negative {
                        return fail("regular triangulation: an infinite cell is not negative");
                    }
                }
            }
        }
        if infinite == 0 {
            return fail("regular triangulation: no infinite cells");
        }
        Ok(())
    }

    fn debug_check(&self) -> Result<()> {
        if cfg!(debug_assertions) {
            self.check()
        } else {
            Ok(())
        }
    }

    /// The finite cells, as vertex lists, and the hidden points.
    pub(crate) fn finish(self) -> (Vec<Vec<u32>>, Vec<u32>) {
        let cells = self
            .tds
            .cells()
            .filter(|&c| self.tds.infinite_slot(c).is_none())
            .map(|c| self.tds.verts(c).to_vec())
            .collect();
        (cells, self.hidden)
    }
}
