//! The compatibility table: which pairs of simplices may lie in one
//! triangulation.

use std::fmt;

use super::bits;
use crate::error::{Error, Result};
use crate::om::{Circuit, OrientedMatroid, circuits};
use crate::simplex::{Simplex, SimplexSet, combinations};

/// The *universe* of simplices a triangulation may use, and, for each, the
/// simplices of the universe it is compatible with.
///
/// The universe is every basis of the oriented matroid, or with
/// `only_empty` every *empty* basis: one that contains no other element of
/// the configuration (TOPCOM.md section 7.7). Two simplices are compatible
/// when their relative interiors are disjoint, so that they meet in a common
/// face: exactly when no circuit `(Z⁺, Z⁻)` has `Z⁺` in one and `Z⁻` in the
/// other. A set of pairwise compatible simplices is a partial
/// triangulation.
///
/// This is TOPCOM's `Admissibles` (`Admissibles.cc:66-216`, `TOPCOM.md`
/// section 7.4), with the universe indexed in canonical order — TOPCOM's
/// simplex index order — and each row a bitset over it, so that pruning a
/// partial triangulation's candidates is a word-wise `AND`. The table takes
/// `|universe|² / 8` bytes: 3 MB for 5 000 simplices.
#[derive(Clone, PartialEq, Eq)]
pub struct Admissibles {
    universe: Vec<Simplex>,
    words: usize,
    /// Row `i`, the simplices compatible with `universe[i]`, is
    /// `table[i * words..(i + 1) * words]`.
    table: Vec<u64>,
}

impl Admissibles {
    /// The table for `om`, over every basis or, with `only_empty`, every
    /// empty basis.
    ///
    /// # Errors
    ///
    /// [`Error::EmptyConfiguration`] for rank zero, and whatever
    /// [`OrientedMatroid::sign`] returns.
    pub fn new(om: &impl OrientedMatroid, only_empty: bool) -> Result<Self> {
        let circuits = circuits(om)?;
        Self::from_circuits(om, &circuits, only_empty)
    }

    /// As [`new`](Self::new), from the circuits of `om` already computed by
    /// [`circuits`].
    ///
    /// # Errors
    ///
    /// As [`new`](Self::new).
    pub fn from_circuits(
        om: &impl OrientedMatroid,
        circuits: &[Circuit],
        only_empty: bool,
    ) -> Result<Self> {
        if om.rank() == 0 {
            return Err(Error::EmptyConfiguration);
        }
        let mut universe = Vec::new();
        for basis in combinations(om.n(), om.rank()) {
            if !om.sign(&basis)?.is_zero() {
                universe.push(basis);
            }
        }
        if only_empty {
            // A basis is not empty when some circuit has one side inside it
            // and a single element on the other: that element lies in the
            // relative interior of the face spanned by the first side
            // (`Admissibles.cc:114-135`). With an empty side this marks
            // everything, as TOPCOM's test does; that circuit is a zero
            // vector, which no fine triangulation can use anyway.
            let stars = Stars::new(om.n(), &universe);
            let mut non_empty = vec![0; stars.words];
            for z in circuits {
                if z.negative().len() == 1 {
                    bits::or(&mut non_empty, &stars.star(z.positive()));
                }
                if z.positive().len() == 1 {
                    bits::or(&mut non_empty, &stars.star(z.negative()));
                }
            }
            universe = universe
                .into_iter()
                .enumerate()
                .filter(|&(i, _)| !bits::test(&non_empty, i))
                .map(|(_, s)| s)
                .collect();
        }

        let stars = Stars::new(om.n(), &universe);
        let words = stars.words;
        let mut table = Vec::with_capacity(universe.len() * words);
        for i in 0..universe.len() {
            let mut row = stars.all.clone();
            bits::clear(&mut row, i);
            table.extend_from_slice(&row);
        }
        // Remove every pair a circuit witnesses as overlapping
        // (`Admissibles.cc:191-214`). The relation is symmetric, so each
        // circuit is applied in both directions at once.
        for z in circuits {
            let plus = stars.star(z.positive());
            let minus = stars.star(z.negative());
            for i in bits::ones(&plus) {
                bits::and_not(&mut table[i * words..(i + 1) * words], &minus);
            }
            for i in bits::ones(&minus) {
                bits::and_not(&mut table[i * words..(i + 1) * words], &plus);
            }
        }
        Ok(Self {
            universe,
            words,
            table,
        })
    }

    /// The simplices a triangulation may use, in canonical order.
    pub fn universe(&self) -> &[Simplex] {
        &self.universe
    }

    /// The number of simplices in the universe.
    pub fn len(&self) -> usize {
        self.universe.len()
    }

    /// Whether the universe is empty: no triangulation exists.
    pub fn is_empty(&self) -> bool {
        self.universe.is_empty()
    }

    /// The position of `simplex` in [`universe`](Self::universe).
    pub fn index_of(&self, simplex: &Simplex) -> Option<usize> {
        self.universe.binary_search(simplex).ok()
    }

    /// Whether the simplices at positions `i` and `j` of the universe are
    /// compatible. A simplex is not compatible with itself.
    ///
    /// # Panics
    ///
    /// If either index is out of range.
    pub fn compatible(&self, i: usize, j: usize) -> bool {
        bits::test(self.row(i), j)
    }

    /// The simplices of the universe compatible with `simplex`, TOPCOM's
    /// `Adm[simplex]`; `None` if it is not in the universe.
    pub fn admissible(&self, simplex: &Simplex) -> Option<SimplexSet> {
        let i = self.index_of(simplex)?;
        Some(
            bits::ones(self.row(i))
                .map(|j| self.universe[j].clone())
                .collect(),
        )
    }

    /// The number of `u64` words in a bitset over the universe.
    pub(super) fn words(&self) -> usize {
        self.words
    }

    /// Row `i` of the table, as a bitset over the universe.
    pub(super) fn row(&self, i: usize) -> &[u64] {
        &self.table[i * self.words..(i + 1) * self.words]
    }
}

impl fmt::Debug for Admissibles {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Admissibles")
            .field("universe", &self.universe.len())
            .finish_non_exhaustive()
    }
}

/// For each label, the bitset of universe simplices containing it.
struct Stars {
    words: usize,
    all: Vec<u64>,
    by_vertex: Vec<Vec<u64>>,
}

impl Stars {
    fn new(n: usize, universe: &[Simplex]) -> Self {
        let words = universe.len().div_ceil(64);
        let mut all = vec![0; words];
        let mut by_vertex = vec![vec![0; words]; n];
        for (i, s) in universe.iter().enumerate() {
            bits::set(&mut all, i);
            for v in s {
                bits::set(&mut by_vertex[v], i);
            }
        }
        Self {
            words,
            all,
            by_vertex,
        }
    }

    /// The simplices containing every label of `face`; all of them for the
    /// empty face.
    fn star(&self, face: &Simplex) -> Vec<u64> {
        let mut out = self.all.clone();
        for v in face {
            bits::and(&mut out, &self.by_vertex[v]);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::PointConfig;

    #[test]
    fn the_square_has_two_crossing_pairs() {
        let p = PointConfig::from_row_major(&[0, 0, 1, 0, 1, 1, 0, 1], 4, 2).unwrap();
        let chi = p.chirotope().unwrap();
        let adm = Admissibles::new(&chi, false).unwrap();
        assert_eq!(adm.len(), 4);
        let s = |v: [usize; 3]| Simplex::from(v);
        // The diagonal 0-2 and the diagonal 1-3 cross.
        assert_eq!(
            adm.admissible(&s([0, 1, 2])).unwrap().to_vecs(),
            vec![vec![0, 2, 3]]
        );
        assert_eq!(
            adm.admissible(&s([0, 1, 3])).unwrap().to_vecs(),
            vec![vec![1, 2, 3]]
        );
        for i in 0..4 {
            assert!(!adm.compatible(i, i));
            for j in 0..4 {
                assert_eq!(adm.compatible(i, j), adm.compatible(j, i));
            }
        }
    }

    #[test]
    fn empty_simplices_leave_out_interior_points() {
        // 3 is inside the triangle 0, 1, 2.
        let p = PointConfig::from_row_major(&[0, 0, 3, 0, 0, 3, 1, 1], 4, 2).unwrap();
        let chi = p.chirotope().unwrap();
        let all = Admissibles::new(&chi, false).unwrap();
        assert_eq!(all.len(), 4);
        let big = all.index_of(&Simplex::from([0, 1, 2])).unwrap();
        assert!((0..4).all(|j| !all.compatible(big, j)));
        let empty = Admissibles::new(&chi, true).unwrap();
        assert_eq!(empty.len(), 3);
        assert_eq!(empty.index_of(&Simplex::from([0, 1, 2])), None);
        assert!(empty.admissible(&Simplex::from([0, 1, 3])).unwrap().len() == 2);
    }

    #[test]
    fn duplicates_are_in_no_empty_simplex() {
        let p = PointConfig::from_row_major(&[0, 0, 1, 0, 0, 1, 1, 0], 4, 2).unwrap();
        let chi = p.chirotope().unwrap();
        assert_eq!(Admissibles::new(&chi, false).unwrap().len(), 2);
        assert!(Admissibles::new(&chi, true).unwrap().is_empty());
    }
}
