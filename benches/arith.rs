//! The determinant-sign tiers.
//!
//! ```text
//! cargo bench --bench arith
//! ```

mod common;

use common::Rng;
use criterion::{Criterion, criterion_group, criterion_main};
use dashu_int::IBig;
use triangulumancer::arith::{SquareMatrix, bareiss, filter, sign_of_det, sign_of_det_of};

/// `count` random `n × n` matrices with entries in `-bound..=bound`.
fn matrices(seed: u64, count: usize, n: usize, bound: i64) -> Vec<Vec<i64>> {
    let mut rng = Rng(seed);
    (0..count)
        .map(|_| {
            (0..n * n)
                .map(|_| rng.below(2 * bound as u64 + 1) - bound)
                .collect()
        })
        .collect()
}

/// The same, with the last row a copy of the first: singular, so the filter
/// always declines.
fn singular(seed: u64, count: usize, n: usize, bound: i64) -> Vec<Vec<i64>> {
    let mut ms = matrices(seed, count, n, bound);
    for m in &mut ms {
        let first = m[..n].to_vec();
        m[n * (n - 1)..].copy_from_slice(&first);
    }
    ms
}

fn tiers(c: &mut Criterion) {
    let mut g = c.benchmark_group("sign_of_det");
    // Per call, averaged over 1 000 matrices.
    for n in [3, 4, 6, 8] {
        let generic = matrices(n as u64, 1000, n, 1000);
        let zero = singular(n as u64, 1000, n, 1000);
        let wide = singular(n as u64, 1000, n, i64::MAX / 2);
        g.bench_function(format!("generic_{n}x{n}"), |b| {
            b.iter(|| {
                generic
                    .iter()
                    .map(|m| sign_of_det(m, n).unwrap().as_i8() as i64)
                    .sum::<i64>()
            })
        });
        g.bench_function(format!("filter_only_{n}x{n}"), |b| {
            b.iter(|| {
                generic
                    .iter()
                    .filter_map(|m| filter::sign_of_det(SquareMatrix::new(m, n).unwrap()))
                    .count()
            })
        });
        g.bench_function(format!("singular_{n}x{n}"), |b| {
            b.iter(|| {
                zero.iter()
                    .map(|m| sign_of_det(m, n).unwrap().as_i8() as i64)
                    .sum::<i64>()
            })
        });
        g.bench_function(format!("i128_only_{n}x{n}"), |b| {
            b.iter(|| {
                zero.iter()
                    .filter_map(|m| bareiss::sign_of_det_i128(SquareMatrix::new(m, n).unwrap()))
                    .count()
            })
        });
        g.bench_function(format!("bigint_only_{n}x{n}"), |b| {
            b.iter(|| {
                zero.iter()
                    .map(|m| {
                        bareiss::sign_of_det_bigint(SquareMatrix::new(m, n).unwrap()).as_i8() as i64
                    })
                    .sum::<i64>()
            })
        });
        g.bench_function(format!("singular_wide_{n}x{n}"), |b| {
            b.iter(|| {
                wide.iter()
                    .map(|m| sign_of_det(m, n).unwrap().as_i8() as i64)
                    .sum::<i64>()
            })
        });
    }
    // Lifted Delaunay predicates in high dimension: `IBig` entries, as
    // `regular` builds them for `d >= 10`.
    for n in [8, 12] {
        let big: Vec<Vec<IBig>> = singular(100 + n as u64, 200, n, i64::MAX / 2)
            .into_iter()
            .map(|m| {
                m.into_iter()
                    .map(|x| IBig::from(x) * IBig::from(x))
                    .collect()
            })
            .collect();
        g.bench_function(format!("ibig_singular_{n}x{n}_x200"), |b| {
            b.iter(|| {
                big.iter()
                    .map(|m| sign_of_det_of(m, n).unwrap().as_i8() as i64)
                    .sum::<i64>()
            })
        });
    }
    g.finish();
}

criterion_group!(benches, tiers);
criterion_main!(benches);
