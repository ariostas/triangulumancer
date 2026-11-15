//! Simplices as sets of point labels, and sets of simplices.
//!
//! A [`Simplex`] is a set of indices into a configuration; for a triangulation
//! of a rank-`r` configuration, every maximal simplex has `r` of them. TOPCOM
//! represents these as heap-allocated `IntegerSet`s; here the common case is a
//! single inline `u128` bitmask, so configurations of up to 128 points never
//! allocate for a simplex, and every set operation is a handful of word
//! instructions. Larger configurations still work: a simplex with a label of
//! 128 or more spills to a boxed, sorted list of its labels, so it costs
//! O(cardinality) — O(rank) for a maximal simplex — however large the
//! configuration is. (A bitset would cost O(n) per simplex, and a Delaunay
//! triangulation with O(n) simplices would then cost O(n²).) The
//! representation is canonical (a simplex is inline exactly when all its labels
//! are below 128), so equality and hashing never have to compare across
//! representations.
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
    /// Some label is at least 128: the labels, strictly increasing, so the
    /// representation of a set is unique.
    Spilled(Box<[usize]>),
}

/// A finite set of point labels.
///
/// Cheap to clone for configurations of up to 128 points (a `u128` copy), and
/// O(cardinality) in memory beyond that.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Simplex {
    repr: Repr,
}

impl Default for Simplex {
    fn default() -> Self {
        Self::new()
    }
}

/// Which elements of a sorted merge to keep.
#[derive(Clone, Copy)]
struct Keep {
    only_a: bool,
    only_b: bool,
    both: bool,
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
            Self {
                repr: Repr::Spilled((0..n).collect()),
            }
        }
    }

    /// Canonicalise a strictly increasing label list.
    fn from_sorted(labels: Vec<usize>) -> Self {
        debug_assert!(labels.windows(2).all(|w| w[0] < w[1]));
        match labels.last() {
            Some(&last) if last >= INLINE_BITS => Self {
                repr: Repr::Spilled(labels.into_boxed_slice()),
            },
            _ => Self::from_bits(labels.iter().fold(0, |b, &v| b | 1 << v)),
        }
    }

    /// The labels of a spilled simplex, or `None` for an inline one.
    fn spilled(&self) -> Option<&[usize]> {
        match &self.repr {
            Repr::Inline(_) => None,
            Repr::Spilled(v) => Some(v),
        }
    }

    /// Merge two simplices' sorted label lists, keeping what `keep` says.
    fn merge(&self, other: &Self, keep: Keep) -> Self {
        let (mut a, mut b) = (self.iter().peekable(), other.iter().peekable());
        let mut out = Vec::with_capacity(self.len() + other.len());
        loop {
            match (a.peek(), b.peek()) {
                (None, None) => break,
                (Some(&x), None) => {
                    if !keep.only_a {
                        break;
                    }
                    out.push(x);
                    a.next();
                }
                (None, Some(&y)) => {
                    if !keep.only_b {
                        break;
                    }
                    out.push(y);
                    b.next();
                }
                (Some(&x), Some(&y)) => match x.cmp(&y) {
                    Ordering::Less => {
                        if keep.only_a {
                            out.push(x);
                        }
                        a.next();
                    }
                    Ordering::Greater => {
                        if keep.only_b {
                            out.push(y);
                        }
                        b.next();
                    }
                    Ordering::Equal => {
                        if keep.both {
                            out.push(x);
                        }
                        a.next();
                        b.next();
                    }
                },
            }
        }
        Self::from_sorted(out)
    }

    /// Whether `label` is in the simplex.
    #[inline]
    pub fn contains(&self, label: usize) -> bool {
        match &self.repr {
            Repr::Inline(b) => label < INLINE_BITS && b & (1 << label) != 0,
            Repr::Spilled(v) => v.binary_search(&label).is_ok(),
        }
    }

    /// Add `label`. Returns whether it was newly added.
    pub fn insert(&mut self, label: usize) -> bool {
        match &mut self.repr {
            Repr::Inline(b) if label < INLINE_BITS => {
                let fresh = *b & (1 << label) == 0;
                *b |= 1 << label;
                fresh
            }
            Repr::Inline(_) => {
                let mut v = self.to_vec();
                v.push(label);
                *self = Self::from_sorted(v);
                true
            }
            Repr::Spilled(v) => match v.binary_search(&label) {
                Ok(_) => false,
                Err(i) => {
                    let mut w = Vec::with_capacity(v.len() + 1);
                    w.extend_from_slice(&v[..i]);
                    w.push(label);
                    w.extend_from_slice(&v[i..]);
                    *v = w.into_boxed_slice();
                    true
                }
            },
        }
    }

    /// Remove `label`. Returns whether it was present.
    pub fn remove(&mut self, label: usize) -> bool {
        match &mut self.repr {
            Repr::Inline(b) => {
                let present = label < INLINE_BITS && *b & (1 << label) != 0;
                if present {
                    *b &= !(1 << label);
                }
                present
            }
            Repr::Spilled(v) => match v.binary_search(&label) {
                Err(_) => false,
                Ok(i) => {
                    let mut w = v.to_vec();
                    w.remove(i);
                    *self = Self::from_sorted(w);
                    true
                }
            },
        }
    }

    /// The number of labels; the cardinality, not the dimension.
    pub fn len(&self) -> usize {
        match &self.repr {
            Repr::Inline(b) => b.count_ones() as usize,
            Repr::Spilled(v) => v.len(),
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
        match &self.repr {
            Repr::Inline(0) => None,
            Repr::Inline(b) => Some(127 - b.leading_zeros() as usize),
            Repr::Spilled(v) => v.last().copied(),
        }
    }

    /// The labels in increasing order.
    pub fn iter(&self) -> Iter<'_> {
        match &self.repr {
            Repr::Inline(b) => Iter(IterRepr::Inline(*b)),
            Repr::Spilled(v) => Iter(IterRepr::Spilled(v.iter())),
        }
    }

    /// The labels in increasing order, collected.
    pub fn to_vec(&self) -> Vec<usize> {
        match &self.repr {
            Repr::Inline(_) => self.iter().collect(),
            Repr::Spilled(v) => v.to_vec(),
        }
    }

    /// Whether every label of `self` is in `other`.
    pub fn is_subset(&self, other: &Self) -> bool {
        match (&self.repr, &other.repr) {
            (Repr::Inline(a), Repr::Inline(b)) => a & !b == 0,
            // A spilled set has a label of 128 or more, which no inline set has.
            (Repr::Spilled(_), Repr::Inline(_)) => false,
            _ => self.len() <= other.len() && self.iter().all(|v| other.contains(v)),
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
                let (small, large) = if self.len() <= other.len() {
                    (self, other)
                } else {
                    (other, self)
                };
                !small.iter().any(|v| large.contains(v))
            }
        }
    }

    /// The labels in either simplex.
    pub fn union(&self, other: &Self) -> Self {
        match (&self.repr, &other.repr) {
            (Repr::Inline(a), Repr::Inline(b)) => Self::from_bits(a | b),
            _ => self.merge(
                other,
                Keep {
                    only_a: true,
                    only_b: true,
                    both: true,
                },
            ),
        }
    }

    /// The labels in both simplices.
    pub fn intersection(&self, other: &Self) -> Self {
        match (&self.repr, &other.repr) {
            (Repr::Inline(a), Repr::Inline(b)) => Self::from_bits(a & b),
            // Only labels below 128 can be common.
            (Repr::Inline(a), Repr::Spilled(_)) | (Repr::Spilled(_), Repr::Inline(a)) => {
                let s = if self.spilled().is_some() {
                    self
                } else {
                    other
                };
                Self::from_bits(a & low_bits(s))
            }
            _ => self.merge(
                other,
                Keep {
                    only_a: false,
                    only_b: false,
                    both: true,
                },
            ),
        }
    }

    /// The labels of `self` not in `other`.
    pub fn difference(&self, other: &Self) -> Self {
        match (&self.repr, &other.repr) {
            (Repr::Inline(a), Repr::Inline(b)) => Self::from_bits(a & !b),
            (Repr::Inline(a), Repr::Spilled(_)) => Self::from_bits(a & !low_bits(other)),
            _ => self.merge(
                other,
                Keep {
                    only_a: true,
                    only_b: false,
                    both: false,
                },
            ),
        }
    }

    /// The labels in exactly one of the two simplices.
    pub fn symmetric_difference(&self, other: &Self) -> Self {
        match (&self.repr, &other.repr) {
            (Repr::Inline(a), Repr::Inline(b)) => Self::from_bits(a ^ b),
            _ => self.merge(
                other,
                Keep {
                    only_a: true,
                    only_b: true,
                    both: false,
                },
            ),
        }
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

/// The labels below 128 of a simplex, as a bitmask.
fn low_bits(s: &Simplex) -> u128 {
    match &s.repr {
        Repr::Inline(b) => *b,
        Repr::Spilled(v) => v
            .iter()
            .take_while(|&&x| x < INLINE_BITS)
            .fold(0, |b, &x| b | 1 << x),
    }
}

/// Iterator over the labels of a [`Simplex`], in increasing order.
#[derive(Debug, Clone)]
pub struct Iter<'a>(IterRepr<'a>);

#[derive(Debug, Clone)]
enum IterRepr<'a> {
    /// The labels not yet yielded.
    Inline(u128),
    Spilled(std::slice::Iter<'a, usize>),
}

impl Iterator for Iter<'_> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        match &mut self.0 {
            IterRepr::Inline(b) => {
                if *b == 0 {
                    return None;
                }
                let bit = b.trailing_zeros() as usize;
                *b &= *b - 1;
                Some(bit)
            }
            IterRepr::Spilled(it) => it.next().copied(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = match &self.0 {
            IterRepr::Inline(b) => b.count_ones() as usize,
            IterRepr::Spilled(it) => it.len(),
        };
        (n, Some(n))
    }
}

impl ExactSizeIterator for Iter<'_> {}

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
        // Labels below 128 go straight into a bitmask; any others are
        // collected and merged once, so building a large set is O(k log k)
        // rather than quadratic.
        let mut low = low_bits(self);
        let mut high: Vec<usize> = Vec::new();
        for v in iter {
            if v < INLINE_BITS {
                low |= 1 << v;
            } else {
                high.push(v);
            }
        }
        if high.is_empty() && self.spilled().is_none() {
            *self = Self::from_bits(low);
            return;
        }
        if let Some(v) = self.spilled() {
            high.extend(v.iter().copied().filter(|&x| x >= INLINE_BITS));
        }
        high.sort_unstable();
        high.dedup();
        let mut labels: Vec<usize> = Self::from_bits(low).iter().collect();
        labels.extend(high);
        *self = Self::from_sorted(labels);
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
    /// For two inline simplices: let `x` be the smallest label in exactly one
    /// of the two sets; below `x` the sorted lists agree. If `x` is in `a`, the
    /// next entry of `a` is `x` while the next entry of `b` is its first label
    /// above `x`, if any: so `a < b` exactly when `b` has a label above `x`,
    /// and otherwise `b` is a prefix of `a`. Symmetrically if `x` is in `b`.
    /// Otherwise the label lists are compared directly.
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
        if let (Repr::Spilled(a), Repr::Spilled(b)) = (&self.repr, &other.repr) {
            return a.cmp(b);
        }
        self.iter().cmp(other.iter())
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
        let mut low = 0u128;
        let mut max = 0;
        for t in &self.simplices {
            match t.as_bits() {
                Some(b) => low |= b,
                None => max = max.max(t.last().unwrap_or(0)),
            }
        }
        if max == 0 {
            return Simplex::from_bits(low);
        }
        // Mark labels in a flat table rather than taking repeated unions, which
        // would cost O(n) per simplex once the support spills.
        let mut seen = vec![false; max + 1];
        for t in &self.simplices {
            if t.as_bits().is_none() {
                for v in t {
                    seen[v] = true;
                }
            }
        }
        for v in Simplex::from_bits(low).iter() {
            seen[v] = true;
        }
        Simplex::from_sorted((0..=max).filter(|&v| seen[v]).collect())
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

// ---------------------------------------------------------------------------
// Enumerating subsets
// ---------------------------------------------------------------------------

/// The `k`-element subsets of `{0, ..., n - 1}`, in canonical (lexicographic)
/// order.
///
/// This is TOPCOM's `Permutation(n, k)` stepped with `lexnext()`, the order in
/// which it enumerates bases (`find_non_deg_basis`) and prints chirotopes.
/// Yields the empty simplex once for `k == 0`, and nothing for `k > n`.
pub fn combinations(n: usize, k: usize) -> Combinations {
    Combinations {
        n,
        current: (k <= n).then(|| (0..k).collect()),
    }
}

/// Iterator returned by [`combinations`].
#[derive(Debug, Clone)]
pub struct Combinations {
    n: usize,
    /// The next subset to yield, as increasing labels; `None` when exhausted.
    current: Option<Vec<usize>>,
}

impl Iterator for Combinations {
    type Item = Simplex;

    fn next(&mut self) -> Option<Simplex> {
        let labels = self.current.as_mut()?;
        let out: Simplex = labels.iter().copied().collect();
        // Advance: bump the rightmost label that is not at its maximum, and
        // reset everything after it to the smallest values above it.
        let k = labels.len();
        match (0..k).rev().find(|&i| labels[i] < self.n - k + i) {
            Some(i) => {
                labels[i] += 1;
                for j in i + 1..k {
                    labels[j] = labels[j - 1] + 1;
                }
            }
            None => self.current = None,
        }
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combinations_are_lexicographic_and_complete() {
        let all: Vec<Vec<usize>> = combinations(5, 3).map(|s| s.to_vec()).collect();
        assert_eq!(all.len(), 10);
        assert_eq!(all[0], vec![0, 1, 2]);
        assert_eq!(all[1], vec![0, 1, 3]);
        assert_eq!(all[9], vec![2, 3, 4]);
        assert!(all.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(combinations(3, 0).collect::<Vec<_>>(), vec![Simplex::new()]);
        assert_eq!(combinations(2, 3).count(), 0);
        assert_eq!(combinations(4, 4).count(), 1);
        assert_eq!(combinations(130, 1).last(), Some(Simplex::from([129])));
    }

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
    fn spilled_simplices_cost_their_cardinality() {
        // An inline simplex is a `u128`; the enum adds only a tag.
        assert!(std::mem::size_of::<Simplex>() <= 32);
        // A simplex with a huge label stores its labels, not a bitset as wide
        // as the configuration: O(rank) words, whatever the labels are.
        for top in [129, 1_000, 1_000_000, usize::MAX / 2] {
            let a = s(&[0, 7, 127, top]);
            match &a.repr {
                Repr::Spilled(v) => assert_eq!(v.len(), 4),
                Repr::Inline(_) => panic!("{top} should spill"),
            }
            let b = s(&[1, 7, top - 1, top]);
            for t in [&a | &b, &a & &b, &a - &b, &a ^ &b] {
                assert!(t.spilled().is_none_or(|v| v.len() <= 8));
            }
            assert_eq!((&a & &b).to_vec(), vec![7, top]);
            assert_eq!((&a - &b).to_vec(), vec![0, 127]);
            assert!((&a - &b).as_bits().is_some());
            assert_eq!(a.facets().count(), 4);
            assert!(a.contains(top) && !a.contains(top - 1));
        }
        // The support of many spilled simplices is built in one pass.
        let k: SimplexSet = (0..1000).map(|i| s(&[i, i + 1, 100_000 + i])).collect();
        let sup = k.support();
        assert_eq!(sup.len(), 2001);
        assert_eq!(sup.first(), Some(0));
        assert_eq!(sup.last(), Some(100_999));
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
