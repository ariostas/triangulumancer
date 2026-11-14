//! Simplices as bitsets of point labels, and sets of simplices.
//!
//! A [`Simplex`] is a set of indices into a configuration; for a triangulation
//! of a rank-`r` configuration, every maximal simplex has `r` of them. TOPCOM
//! represents these as heap-allocated `IntegerSet`s; here the common case is a
//! single inline `u128`, so configurations of up to 128 points never allocate
//! for a simplex, and every set operation is a handful of word instructions.
//! Larger configurations still work: a simplex with a label of 128 or more
//! spills to a boxed word slice. The representation is canonical (a simplex is
//! inline exactly when all its labels are below 128), so equality and hashing
//! never have to compare across representations.
//!
//! **Ordering is canonical** (`PLAN.md` section 5.6): simplices compare
//! lexicographically on their sorted label lists, exactly as Python compares
//! `sorted(simplex)` lists, so a [`SimplexSet`] iterates in the order the
//! fixtures and the Python API use.
//!
//! [`SimplexSet`] is the simplicial complex: TOPCOM's `SimplicialComplex`
//! without the process-global intern table (bug B2 in `PLAN.md`). It provides
//! the operations the triangulation algorithms use — [`support`], [`star`],
//! [`link`], [`join`] and [`insert_boundary`] — with TOPCOM's semantics.
//!
//! [`support`]: SimplexSet::support
//! [`star`]: SimplexSet::star
//! [`link`]: SimplexSet::link
//! [`join`]: SimplexSet::join
//! [`insert_boundary`]: SimplexSet::insert_boundary

use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::fmt;
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Sub, SubAssign};

/// Labels below this are stored inline.
const INLINE_BITS: usize = 128;

#[derive(Clone, PartialEq, Eq, Hash)]
enum Repr {
    /// Every label is below 128.
    Inline(u128),
    /// Some label is at least 128. At least three words, and the last word is
    /// non-zero, so the representation of a set is unique.
    Spilled(Box<[u64]>),
}

/// A finite set of point labels, stored as a bitset.
///
/// Cheap to clone for configurations of up to 128 points (a `u128` copy).
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Simplex {
    repr: Repr,
}

impl Default for Simplex {
    fn default() -> Self {
        Self::new()
    }
}

impl Simplex {
    /// The empty simplex.
    pub const fn new() -> Self {
        Self {
            repr: Repr::Inline(0),
        }
    }

    /// The simplex whose labels are the set bits of `bits`.
    pub const fn from_bits(bits: u128) -> Self {
        Self {
            repr: Repr::Inline(bits),
        }
    }

    /// The labels as a `u128` bitmask, if every label is below 128.
    ///
    /// The fast path for algorithms that want to do their own word arithmetic.
    pub fn as_bits(&self) -> Option<u128> {
        match self.repr {
            Repr::Inline(b) => Some(b),
            Repr::Spilled(_) => None,
        }
    }

    /// The simplex `{0, 1, ..., n - 1}`.
    pub fn full(n: usize) -> Self {
        if n <= INLINE_BITS {
            Self::from_bits(if n == INLINE_BITS {
                u128::MAX
            } else {
                (1u128 << n) - 1
            })
        } else {
            let mut words = vec![u64::MAX; n / 64];
            if n % 64 != 0 {
                words.push((1u64 << (n % 64)) - 1);
            }
            Self::from_words(words)
        }
    }

    /// Canonicalise a little-endian word vector.
    fn from_words(mut words: Vec<u64>) -> Self {
        while words.last() == Some(&0) {
            words.pop();
        }
        match words.len() {
            0 => Self::new(),
            1 => Self::from_bits(words[0] as u128),
            2 => Self::from_bits(words[0] as u128 | (words[1] as u128) << 64),
            _ => Self {
                repr: Repr::Spilled(words.into_boxed_slice()),
            },
        }
    }

    /// The number of 64-bit words in use.
    fn n_words(&self) -> usize {
        match &self.repr {
            Repr::Inline(_) => 2,
            Repr::Spilled(w) => w.len(),
        }
    }

    /// Word `i`, zero past the end.
    #[inline]
    fn word(&self, i: usize) -> u64 {
        match &self.repr {
            Repr::Inline(b) => match i {
                0 => *b as u64,
                1 => (*b >> 64) as u64,
                _ => 0,
            },
            Repr::Spilled(w) => w.get(i).copied().unwrap_or(0),
        }
    }

    /// Combine two simplices word by word.
    #[inline]
    fn zip_with(
        &self,
        other: &Self,
        f: impl Fn(u64, u64) -> u64,
        g: impl Fn(u128, u128) -> u128,
    ) -> Self {
        match (&self.repr, &other.repr) {
            (Repr::Inline(a), Repr::Inline(b)) => Self::from_bits(g(*a, *b)),
            _ => {
                let n = self.n_words().max(other.n_words());
                Self::from_words((0..n).map(|i| f(self.word(i), other.word(i))).collect())
            }
        }
    }

    /// Whether `label` is in the simplex.
    #[inline]
    pub fn contains(&self, label: usize) -> bool {
        match &self.repr {
            Repr::Inline(b) => label < INLINE_BITS && b & (1 << label) != 0,
            Repr::Spilled(_) => self.word(label / 64) & (1 << (label % 64)) != 0,
        }
    }

    /// Add `label`. Returns whether it was newly added.
    pub fn insert(&mut self, label: usize) -> bool {
        if self.contains(label) {
            return false;
        }
        match &mut self.repr {
            Repr::Inline(b) if label < INLINE_BITS => *b |= 1 << label,
            _ => {
                let n = self.n_words().max(label / 64 + 1);
                let mut words: Vec<u64> = (0..n).map(|i| self.word(i)).collect();
                words[label / 64] |= 1 << (label % 64);
                *self = Self::from_words(words);
            }
        }
        true
    }

    /// Remove `label`. Returns whether it was present.
    pub fn remove(&mut self, label: usize) -> bool {
        if !self.contains(label) {
            return false;
        }
        match &mut self.repr {
            Repr::Inline(b) => *b &= !(1 << label),
            Repr::Spilled(w) => {
                let mut words = w.to_vec();
                words[label / 64] &= !(1 << (label % 64));
                *self = Self::from_words(words);
            }
        }
        true
    }

    /// The number of labels; the cardinality, not the dimension.
    pub fn len(&self) -> usize {
        match &self.repr {
            Repr::Inline(b) => b.count_ones() as usize,
            Repr::Spilled(w) => w.iter().map(|x| x.count_ones() as usize).sum(),
        }
    }

    /// Whether the simplex has no labels.
    pub fn is_empty(&self) -> bool {
        matches!(self.repr, Repr::Inline(0))
    }

    /// The smallest label.
    pub fn first(&self) -> Option<usize> {
        self.iter().next()
    }

    /// The largest label.
    pub fn last(&self) -> Option<usize> {
        let n = self.n_words();
        (0..n)
            .rev()
            .find(|&i| self.word(i) != 0)
            .map(|i| i * 64 + 63 - self.word(i).leading_zeros() as usize)
    }

    /// The labels in increasing order.
    pub fn iter(&self) -> Iter<'_> {
        Iter {
            simplex: self,
            index: 0,
            current: self.word(0),
        }
    }

    /// The labels in increasing order, collected.
    pub fn to_vec(&self) -> Vec<usize> {
        self.iter().collect()
    }

    /// Whether every label of `self` is in `other`.
    pub fn is_subset(&self, other: &Self) -> bool {
        match (&self.repr, &other.repr) {
            (Repr::Inline(a), Repr::Inline(b)) => a & !b == 0,
            _ => (0..self.n_words()).all(|i| self.word(i) & !other.word(i) == 0),
        }
    }

    /// Whether every label of `other` is in `self`.
    pub fn is_superset(&self, other: &Self) -> bool {
        other.is_subset(self)
    }

    /// Whether the two simplices share no label.
    pub fn is_disjoint(&self, other: &Self) -> bool {
        match (&self.repr, &other.repr) {
            (Repr::Inline(a), Repr::Inline(b)) => a & b == 0,
            _ => {
                (0..self.n_words().min(other.n_words())).all(|i| self.word(i) & other.word(i) == 0)
            }
        }
    }

    /// The labels in either simplex.
    pub fn union(&self, other: &Self) -> Self {
        self.zip_with(other, |a, b| a | b, |a, b| a | b)
    }

    /// The labels in both simplices.
    pub fn intersection(&self, other: &Self) -> Self {
        self.zip_with(other, |a, b| a & b, |a, b| a & b)
    }

    /// The labels of `self` not in `other`.
    pub fn difference(&self, other: &Self) -> Self {
        self.zip_with(other, |a, b| a & !b, |a, b| a & !b)
    }

    /// The labels in exactly one of the two simplices.
    pub fn symmetric_difference(&self, other: &Self) -> Self {
        self.zip_with(other, |a, b| a ^ b, |a, b| a ^ b)
    }

    /// The facets: `self` minus one label, in increasing order of the label
    /// removed. Empty for the empty simplex.
    pub fn facets(&self) -> impl Iterator<Item = Simplex> + '_ {
        self.iter().map(move |v| {
            let mut f = self.clone();
            f.remove(v);
            f
        })
    }
}

/// Iterator over the labels of a [`Simplex`], in increasing order.
#[derive(Debug, Clone)]
pub struct Iter<'a> {
    simplex: &'a Simplex,
    index: usize,
    current: u64,
}

impl Iterator for Iter<'_> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        while self.current == 0 {
            self.index += 1;
            if self.index >= self.simplex.n_words() {
                return None;
            }
            self.current = self.simplex.word(self.index);
        }
        let bit = self.current.trailing_zeros() as usize;
        self.current &= self.current - 1;
        Some(self.index * 64 + bit)
    }
}

impl<'a> IntoIterator for &'a Simplex {
    type Item = usize;
    type IntoIter = Iter<'a>;

    fn into_iter(self) -> Iter<'a> {
        self.iter()
    }
}

impl FromIterator<usize> for Simplex {
    fn from_iter<I: IntoIterator<Item = usize>>(iter: I) -> Self {
        let mut s = Self::new();
        s.extend(iter);
        s
    }
}

impl Extend<usize> for Simplex {
    fn extend<I: IntoIterator<Item = usize>>(&mut self, iter: I) {
        for v in iter {
            self.insert(v);
        }
    }
}

impl From<&[usize]> for Simplex {
    fn from(labels: &[usize]) -> Self {
        labels.iter().copied().collect()
    }
}

impl<const N: usize> From<[usize; N]> for Simplex {
    fn from(labels: [usize; N]) -> Self {
        labels.into_iter().collect()
    }
}

impl Ord for Simplex {
    /// Lexicographic order on the sorted label lists, a proper prefix first.
    ///
    /// Let `x` be the smallest label in exactly one of the two sets; below `x`
    /// the sorted lists agree. If `x` is in `a`, the next entry of `a` is `x`
    /// while the next entry of `b` is its first label above `x`, if any: so
    /// `a < b` exactly when `b` has a label above `x`, and otherwise `b` is a
    /// prefix of `a`. Symmetrically if `x` is in `b`.
    fn cmp(&self, other: &Self) -> Ordering {
        if let (Repr::Inline(a), Repr::Inline(b)) = (&self.repr, &other.repr) {
            let diff = a ^ b;
            if diff == 0 {
                return Ordering::Equal;
            }
            let x = diff.trailing_zeros();
            // Labels strictly above `x`; shifting in two steps avoids a shift
            // by 128 when `x == 127`.
            let above = |s: u128| (s >> x) >> 1 != 0;
            return if a & (1 << x) != 0 {
                if above(*b) {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            } else if above(*a) {
                Ordering::Greater
            } else {
                Ordering::Less
            };
        }
        let n = self.n_words().max(other.n_words());
        let Some(i) = (0..n).find(|&i| self.word(i) != other.word(i)) else {
            return Ordering::Equal;
        };
        let (wa, wb) = (self.word(i), other.word(i));
        let bit = (wa ^ wb).trailing_zeros();
        let above =
            |s: &Simplex, w: u64| (w >> bit) >> 1 != 0 || (i + 1..n).any(|j| s.word(j) != 0);
        if wa & (1 << bit) != 0 {
            if above(other, wb) {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        } else if above(self, wa) {
            Ordering::Greater
        } else {
            Ordering::Less
        }
    }
}

impl PartialOrd for Simplex {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Debug for Simplex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

impl fmt::Display for Simplex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

macro_rules! set_ops {
    ($($op:ident, $method:ident, $assign:ident, $assign_method:ident, $via:ident;)*) => {$(
        impl $op<&Simplex> for &Simplex {
            type Output = Simplex;
            fn $method(self, rhs: &Simplex) -> Simplex {
                self.$via(rhs)
            }
        }
        impl $assign<&Simplex> for Simplex {
            fn $assign_method(&mut self, rhs: &Simplex) {
                *self = self.$via(rhs);
            }
        }
    )*};
}

set_ops! {
    BitOr, bitor, BitOrAssign, bitor_assign, union;
    BitAnd, bitand, BitAndAssign, bitand_assign, intersection;
    Sub, sub, SubAssign, sub_assign, difference;
    BitXor, bitxor, BitXorAssign, bitxor_assign, symmetric_difference;
}

// ---------------------------------------------------------------------------
// Sets of simplices
// ---------------------------------------------------------------------------

/// A finite set of simplices: a simplicial complex given by its listed faces
/// (TOPCOM's `SimplicialComplex`), or a triangulation given by its maximal
/// simplices.
///
/// Iterates in canonical order. Equality, ordering and hashing are those of the
/// underlying sorted set, so two triangulations with the same simplices are
/// equal however they were built.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SimplexSet {
    simplices: BTreeSet<Simplex>,
}

impl SimplexSet {
    /// The empty set: no simplices at all, not even the empty one.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a simplex. Returns whether it was newly added.
    pub fn insert(&mut self, simplex: Simplex) -> bool {
        self.simplices.insert(simplex)
    }

    /// Remove a simplex. Returns whether it was present.
    pub fn remove(&mut self, simplex: &Simplex) -> bool {
        self.simplices.remove(simplex)
    }

    /// Whether `simplex` is one of the listed simplices.
    pub fn contains(&self, simplex: &Simplex) -> bool {
        self.simplices.contains(simplex)
    }

    /// The number of simplices.
    pub fn len(&self) -> usize {
        self.simplices.len()
    }

    /// Whether there are no simplices.
    pub fn is_empty(&self) -> bool {
        self.simplices.is_empty()
    }

    /// The simplices, in canonical order.
    pub fn iter(&self) -> std::collections::btree_set::Iter<'_, Simplex> {
        self.simplices.iter()
    }

    /// The simplices as sorted label lists, in canonical order: the form the
    /// Python API returns and the fixtures record.
    pub fn to_vecs(&self) -> Vec<Vec<usize>> {
        self.simplices.iter().map(Simplex::to_vec).collect()
    }

    /// The common cardinality of all simplices, if they share one. `None` for
    /// an empty set or a mixed one.
    pub fn pure_cardinality(&self) -> Option<usize> {
        let mut it = self.simplices.iter().map(Simplex::len);
        let first = it.next()?;
        it.all(|c| c == first).then_some(first)
    }

    /// The union of all simplices: the labels used. A triangulation is *fine*
    /// exactly when its support is every label of the configuration.
    pub fn support(&self) -> Simplex {
        let mut s = Simplex::new();
        for t in &self.simplices {
            s |= t;
        }
        s
    }

    /// The simplices containing `simplex` (itself included, if present).
    pub fn star(&self, simplex: &Simplex) -> Self {
        self.simplices
            .iter()
            .filter(|t| simplex.is_subset(t))
            .cloned()
            .collect()
    }

    /// `{ t \ simplex : t in self, simplex ⊆ t }`.
    ///
    /// Contains the empty simplex exactly when `simplex` itself is listed. This
    /// is TOPCOM's `SimplicialComplex::link`.
    pub fn link(&self, simplex: &Simplex) -> Self {
        self.simplices
            .iter()
            .filter(|t| simplex.is_subset(t))
            .map(|t| t - simplex)
            .collect()
    }

    /// `{ t ∪ simplex : t in self }`.
    ///
    /// The join is only a simplicial join when `simplex` is disjoint from every
    /// listed simplex, as TOPCOM assumes (and checks only in debug builds); the
    /// union is taken regardless.
    pub fn join_simplex(&self, simplex: &Simplex) -> Self {
        self.simplices.iter().map(|t| t | simplex).collect()
    }

    /// `{ s ∪ t : s in self, t in other }`.
    ///
    /// Joining with the empty set gives the empty set, and joining with the set
    /// holding only the empty simplex is the identity, as in TOPCOM.
    pub fn join(&self, other: &Self) -> Self {
        let mut out = Self::new();
        for s in &self.simplices {
            for t in &other.simplices {
                out.insert(s | t);
            }
        }
        out
    }

    /// Add every facet of `simplex`. Adds nothing for the empty simplex.
    pub fn insert_boundary(&mut self, simplex: &Simplex) {
        for f in simplex.facets() {
            self.insert(f);
        }
    }

    /// The facets of `simplex`, as a set.
    pub fn boundary(simplex: &Simplex) -> Self {
        let mut out = Self::new();
        out.insert_boundary(simplex);
        out
    }

    /// The simplices in either set.
    pub fn union(&self, other: &Self) -> Self {
        self.simplices.union(&other.simplices).cloned().collect()
    }

    /// The simplices in both sets.
    pub fn intersection(&self, other: &Self) -> Self {
        self.simplices
            .intersection(&other.simplices)
            .cloned()
            .collect()
    }

    /// The simplices of `self` not in `other`.
    pub fn difference(&self, other: &Self) -> Self {
        self.simplices
            .difference(&other.simplices)
            .cloned()
            .collect()
    }
}

impl FromIterator<Simplex> for SimplexSet {
    fn from_iter<I: IntoIterator<Item = Simplex>>(iter: I) -> Self {
        Self {
            simplices: iter.into_iter().collect(),
        }
    }
}

impl Extend<Simplex> for SimplexSet {
    fn extend<I: IntoIterator<Item = Simplex>>(&mut self, iter: I) {
        self.simplices.extend(iter);
    }
}

impl<'a> IntoIterator for &'a SimplexSet {
    type Item = &'a Simplex;
    type IntoIter = std::collections::btree_set::Iter<'a, Simplex>;

    fn into_iter(self) -> Self::IntoIter {
        self.simplices.iter()
    }
}

impl IntoIterator for SimplexSet {
    type Item = Simplex;
    type IntoIter = std::collections::btree_set::IntoIter<Simplex>;

    fn into_iter(self) -> Self::IntoIter {
        self.simplices.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(labels: &[usize]) -> Simplex {
        Simplex::from(labels)
    }

    /// A deterministic spread of label sets, inline and spilled.
    fn samples() -> Vec<Vec<usize>> {
        let mut out = vec![
            vec![],
            vec![0],
            vec![1],
            vec![0, 1],
            vec![0, 2],
            vec![1, 2],
            vec![0, 1, 2],
            vec![63],
            vec![64],
            vec![63, 64],
            vec![127],
            vec![0, 127],
            vec![128],
            vec![0, 128],
            vec![127, 128],
            vec![5, 200],
            vec![5, 199, 200],
            vec![300],
            vec![0, 1, 300],
        ];
        let mut x: u64 = 12345;
        for _ in 0..200 {
            let mut v = Vec::new();
            for _ in 0..(x % 6) {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let bound = if x % 3 == 0 { 260 } else { 10 };
                v.push(((x >> 33) % bound) as usize);
            }
            v.sort_unstable();
            v.dedup();
            out.push(v);
        }
        out
    }

    #[test]
    fn round_trip_and_canonical_representation() {
        for v in samples() {
            let a = s(&v);
            assert_eq!(a.to_vec(), v);
            assert_eq!(a.len(), v.len());
            assert_eq!(a.is_empty(), v.is_empty());
            assert_eq!(a.first(), v.first().copied());
            assert_eq!(a.last(), v.last().copied());
            assert_eq!(a.as_bits().is_some(), v.iter().all(|&x| x < 128));
            // Building the same set another way gives an identical value.
            let b: Simplex = v.iter().rev().copied().collect();
            assert_eq!(a, b);
        }
        // Removing the only high label returns to the inline representation.
        let mut a = s(&[3, 500]);
        a.remove(500);
        assert_eq!(a, s(&[3]));
        assert!(a.as_bits().is_some());
    }

    #[test]
    fn ordering_is_lexicographic_on_sorted_lists() {
        let all = samples();
        for x in &all {
            for y in &all {
                assert_eq!(s(x).cmp(&s(y)), x.cmp(y), "{x:?} vs {y:?}");
            }
        }
    }

    #[test]
    fn set_operations_match_btreeset() {
        let all = samples();
        for x in all.iter().step_by(3) {
            for y in all.iter().step_by(5) {
                let (a, b) = (s(x), s(y));
                let (bx, by): (BTreeSet<usize>, BTreeSet<usize>) =
                    (x.iter().copied().collect(), y.iter().copied().collect());
                let v = |set: BTreeSet<usize>| set.into_iter().collect::<Vec<_>>();
                assert_eq!((&a | &b).to_vec(), v(&bx | &by));
                assert_eq!((&a & &b).to_vec(), v(&bx & &by));
                assert_eq!((&a - &b).to_vec(), v(&bx - &by));
                assert_eq!((&a ^ &b).to_vec(), v(&bx ^ &by));
                assert_eq!(a.is_subset(&b), bx.is_subset(&by));
                assert_eq!(a.is_superset(&b), bx.is_superset(&by));
                assert_eq!(a.is_disjoint(&b), bx.is_disjoint(&by));
            }
        }
    }

    #[test]
    fn insert_remove_contains() {
        let mut a = Simplex::new();
        assert!(a.insert(5));
        assert!(!a.insert(5));
        assert!(a.insert(130));
        assert!(a.contains(5) && a.contains(130) && !a.contains(6) && !a.contains(10_000));
        assert!(a.remove(130));
        assert!(!a.remove(130));
        assert_eq!(a, s(&[5]));
        assert_eq!(Simplex::full(3), s(&[0, 1, 2]));
        assert_eq!(Simplex::full(128).len(), 128);
        assert_eq!(Simplex::full(130).to_vec(), (0..130).collect::<Vec<_>>());
        assert_eq!(Simplex::full(0), Simplex::new());
    }

    #[test]
    fn facets() {
        let f: Vec<_> = s(&[1, 4, 7]).facets().map(|f| f.to_vec()).collect();
        assert_eq!(f, vec![vec![4, 7], vec![1, 7], vec![1, 4]]);
        assert_eq!(Simplex::new().facets().count(), 0);
    }

    #[test]
    fn complex_operations() {
        // Two triangles sharing the edge {1, 2}.
        let k: SimplexSet = [s(&[0, 1, 2]), s(&[1, 2, 3])].into_iter().collect();
        assert_eq!(k.support(), s(&[0, 1, 2, 3]));
        assert_eq!(k.pure_cardinality(), Some(3));
        assert_eq!(k.link(&s(&[1, 2])).to_vecs(), vec![vec![0], vec![3]]);
        assert_eq!(k.link(&s(&[0])).to_vecs(), vec![vec![1, 2]]);
        assert_eq!(k.link(&s(&[0, 3])), SimplexSet::new());
        // A listed simplex has the empty simplex as its link.
        assert_eq!(k.link(&s(&[0, 1, 2])).to_vecs(), vec![Vec::<usize>::new()]);
        assert_eq!(k.star(&s(&[3])).to_vecs(), vec![vec![1, 2, 3]]);
        assert_eq!(
            k.join_simplex(&s(&[9])).to_vecs(),
            vec![vec![0, 1, 2, 9], vec![1, 2, 3, 9]]
        );

        let empty = SimplexSet::new();
        let unit: SimplexSet = [Simplex::new()].into_iter().collect();
        assert_eq!(k.join(&empty), empty);
        assert_eq!(k.join(&unit), k);
        let pts: SimplexSet = [s(&[8]), s(&[9])].into_iter().collect();
        assert_eq!(k.join(&pts).len(), 4);

        let mut b = SimplexSet::new();
        b.insert_boundary(&s(&[0, 1, 2]));
        assert_eq!(b.to_vecs(), vec![vec![0, 1], vec![0, 2], vec![1, 2]]);
        assert_eq!(b, SimplexSet::boundary(&s(&[0, 1, 2])));
        assert_eq!(b.pure_cardinality(), Some(2));
        b.insert(s(&[5]));
        assert_eq!(b.pure_cardinality(), None);
    }

    #[test]
    fn canonical_iteration_order() {
        let k: SimplexSet = [s(&[1, 2, 3]), s(&[0, 2, 3]), s(&[0, 1, 3]), s(&[0, 1, 2])]
            .into_iter()
            .collect();
        assert_eq!(
            k.to_vecs(),
            vec![vec![0, 1, 2], vec![0, 1, 3], vec![0, 2, 3], vec![1, 2, 3]]
        );
    }
}
