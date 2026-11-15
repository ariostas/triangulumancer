//! Delaunay and regular triangulations.
//!
//! ```text
//! cargo bench --bench regular
//! ```

mod common;

use common::{random_points, random_weights};
use criterion::{Criterion, criterion_group, criterion_main};
use triangulumancer::regular::{self, Scalar};

fn delaunay(c: &mut Criterion) {
    let mut g = c.benchmark_group("delaunay");
    g.sample_size(10);
    for (name, seed, n, dim) in [
        ("random_10000x2", 10, 10_000, 2),
        ("random_2000x3", 11, 2_000, 3),
        ("random_300x5", 12, 300, 5),
        ("random_40x10", 13, 40, 10),
    ] {
        let config = random_points(seed, n, dim, 1 << 20);
        g.bench_function(name, |b| b.iter(|| regular::delaunay(&config).unwrap()));
    }
    // A lattice: maximally cospherical, every tie broken symbolically.
    let lattice = random_points(14, 2_000, 3, 8);
    g.bench_function("lattice_2000x3_range8", |b| {
        b.iter(|| regular::delaunay(&lattice).unwrap())
    });
    g.finish();
}

fn weights(c: &mut Criterion) {
    let mut g = c.benchmark_group("regular");
    g.sample_size(10);
    let config = random_points(11, 2_000, 3, 1 << 20);
    let weights: Vec<Scalar> = random_weights(11, 2_000, 1e10)
        .into_iter()
        .map(Scalar::Float)
        .collect();
    g.bench_function("float_weights_2000x3", |b| {
        b.iter(|| regular::with_weights(&config, &weights).unwrap())
    });
    g.finish();
}

criterion_group!(benches, delaunay, weights);
criterion_main!(benches);
