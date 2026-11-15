//! A fast, non-cryptographic hasher for the crate's internal hash tables.
//!
//! The standard library's SipHash resists hash flooding, which matters for
//! tables keyed by untrusted input. Here every table is keyed by simplices or
//! triangulations the algorithms built themselves, and SipHash was a tenth of
//! the flip graph's run time. This is the multiply-rotate scheme of
//! `rustc-hash` (the compiler's own `FxHasher`), a dozen lines not worth a
//! dependency.

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};

/// A `HashMap` with [`FastHasher`].
pub(crate) type FastHashMap<K, V> = HashMap<K, V, BuildHasherDefault<FastHasher>>;

/// A `HashSet` with [`FastHasher`].
pub(crate) type FastHashSet<T> = HashSet<T, BuildHasherDefault<FastHasher>>;

/// One multiply per word; see the [module documentation](self).
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct FastHasher {
    hash: u64,
}

/// An odd constant with well-spread bits (from `rustc-hash` 2).
const K: u64 = 0xf135_7aea_2e62_a9c5;

impl FastHasher {
    #[inline]
    fn add(&mut self, word: u64) {
        self.hash = self.hash.wrapping_add(word).wrapping_mul(K);
    }
}

impl Hasher for FastHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut chunks = bytes.chunks_exact(8);
        for c in &mut chunks {
            self.add(u64::from_le_bytes(c.try_into().unwrap_or_default()));
        }
        let rest = chunks.remainder();
        if !rest.is_empty() {
            let mut buf = [0u8; 8];
            buf[..rest.len()].copy_from_slice(rest);
            self.add(u64::from_le_bytes(buf));
        }
    }

    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.add(u64::from(i));
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.add(u64::from(i));
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }

    #[inline]
    fn write_u128(&mut self, i: u128) {
        self.add(i as u64);
        self.add((i >> 64) as u64);
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }

    #[inline]
    fn finish(&self) -> u64 {
        // The multiply leaves the low bits weakest, and hashbrown takes its
        // bucket index from them: rotate the strong high bits down.
        self.hash.rotate_left(26)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simplex::{Simplex, combinations};

    #[test]
    fn distinguishes_simplices() {
        let all: FastHashSet<Simplex> = combinations(20, 4).collect();
        assert_eq!(all.len(), 4845);
        let big: FastHashSet<Simplex> = (0..1000).map(|i| Simplex::from([i, 200 + i])).collect();
        assert_eq!(big.len(), 1000);
    }
}
