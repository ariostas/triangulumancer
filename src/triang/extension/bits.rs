//! Fixed-width bitsets as `u64` slices, for the extension graph's tables.

#[inline]
pub(super) fn set(bits: &mut [u64], i: usize) {
    bits[i / 64] |= 1 << (i % 64);
}

#[inline]
pub(super) fn clear(bits: &mut [u64], i: usize) {
    bits[i / 64] &= !(1 << (i % 64));
}

#[inline]
pub(super) fn flip(bits: &mut [u64], i: usize) {
    bits[i / 64] ^= 1 << (i % 64);
}

#[inline]
pub(super) fn test(bits: &[u64], i: usize) -> bool {
    bits[i / 64] & (1 << (i % 64)) != 0
}

pub(super) fn or(a: &mut [u64], b: &[u64]) {
    a.iter_mut().zip(b).for_each(|(x, y)| *x |= y);
}

pub(super) fn and(a: &mut [u64], b: &[u64]) {
    a.iter_mut().zip(b).for_each(|(x, y)| *x &= y);
}

pub(super) fn and_not(a: &mut [u64], b: &[u64]) {
    a.iter_mut().zip(b).for_each(|(x, y)| *x &= !y);
}

/// The smallest set bit at position `from` or later.
#[inline]
pub(super) fn next(bits: &[u64], from: usize) -> Option<usize> {
    let mut w = from / 64;
    if w >= bits.len() {
        return None;
    }
    let mut word = bits[w] & (u64::MAX << (from % 64));
    loop {
        if word != 0 {
            return Some(w * 64 + word.trailing_zeros() as usize);
        }
        w += 1;
        word = *bits.get(w)?;
    }
}

/// The set bits, in increasing order.
pub(super) fn ones(bits: &[u64]) -> impl Iterator<Item = usize> + '_ {
    bits.iter().enumerate().flat_map(|(w, &word)| {
        let mut word = word;
        std::iter::from_fn(move || {
            (word != 0).then(|| {
                let i = word.trailing_zeros() as usize;
                word &= word - 1;
                w * 64 + i
            })
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn next_and_ones() {
        let mut b = vec![0; 3];
        for i in [0, 5, 63, 64, 130] {
            set(&mut b, i);
        }
        assert_eq!(ones(&b).collect::<Vec<_>>(), vec![0, 5, 63, 64, 130]);
        assert_eq!(next(&b, 0), Some(0));
        assert_eq!(next(&b, 1), Some(5));
        assert_eq!(next(&b, 64), Some(64));
        assert_eq!(next(&b, 65), Some(130));
        assert_eq!(next(&b, 131), None);
        assert_eq!(next(&b, 500), None);
        clear(&mut b, 5);
        flip(&mut b, 63);
        assert!(!test(&b, 5) && !test(&b, 63) && test(&b, 64));
        assert_eq!(next(&[], 0), None);
    }
}
