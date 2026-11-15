//! Placing and fine triangulations, flips, and the two enumerations.
//!
//! ```text
//! cargo bench --bench triang
//! ```

mod common;

use std::collections::HashSet;
use std::hint::black_box;
use std::time::Duration;

use common::{grid, random_points, vc_degenerate_flip};
use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use triangulumancer::triang::{
    ExtensionGraph, FlipGraph, Triangulation, fine, flip::flip_reps, neighbors, placing,
};

fn construction(c: &mut Criterion) {
    let mut g = c.benchmark_group("construction");
    g.sample_size(20);
    for (name, config) in [
        ("random_100x3", random_points(1, 100, 3, 1000)),
        ("random_60x4", random_points(2, 60, 4, 1000)),
        // Labels past 128: spilled simplices.
        ("random_300x3", random_points(4, 300, 3, 10_000)),
    ] {
        g.bench_function(format!("placing/{name}"), |b| {
            b.iter(|| placing(&config.chirotope().unwrap()).unwrap())
        });
        g.bench_function(format!("fine/{name}"), |b| {
            b.iter(|| fine(&config.chirotope().unwrap()).unwrap())
        });
    }
    g.finish();
}

fn flips(c: &mut Criterion) {
    let mut g = c.benchmark_group("flips");
    let config = random_points(3, 40, 3, 100);
    let chi = config.chirotope().unwrap();
    let t = fine(&chi).unwrap();
    // A warm chirotope: what the flip graph sees after its first few nodes.
    g.bench_function("flip_reps/random_40x3_warm", |b| {
        b.iter(|| flip_reps(&chi, &t, false).unwrap())
    });
    // A cold one each time: what `Triangulation.neighbors()` costs in Python.
    g.bench_function("neighbors/random_40x3_cold", |b| {
        b.iter(|| neighbors(&config.chirotope().unwrap(), &t).unwrap())
    });
    g.finish();
}

fn flip_graph(c: &mut Criterion) {
    let mut g = c.benchmark_group("flip_graph");
    g.sample_size(10).measurement_time(Duration::from_secs(10));
    let config = grid(3, 4);
    for only_fine in [false, true] {
        let name = if only_fine { "fine" } else { "all" };
        g.bench_function(format!("bfs_{name}/grid_3x4"), |b| {
            b.iter(|| {
                let chi = config.chirotope().unwrap();
                FlipGraph::new(&chi, only_fine).unwrap().count()
            })
        });
    }
    g.finish();
}

fn extension(c: &mut Criterion) {
    let mut g = c.benchmark_group("extension");
    g.sample_size(20);
    let config = grid(3, 4);
    let chi = config.chirotope().unwrap();
    g.bench_function("count/grid_3x4", |b| {
        b.iter(|| ExtensionGraph::new(&chi, false).unwrap().count())
    });
    g.bench_function("iterate/grid_3x4", |b| {
        b.iter(|| {
            ExtensionGraph::new(&chi, false)
                .unwrap()
                .map(black_box)
                .fold(0, |n, _| n + 1)
        })
    });
    let vc = vc_degenerate_flip();
    let chi = vc.chirotope().unwrap();
    g.bench_function("setup/vc_degenerate_flip", |b| {
        b.iter(|| ExtensionGraph::new(&vc.chirotope().unwrap(), false).unwrap())
    });
    g.bench_function("iterate_100k/vc_degenerate_flip", |b| {
        b.iter_batched(
            || ExtensionGraph::new(&chi, false).unwrap(),
            |graph| graph.take(100_000).map(black_box).fold(0, |n, _| n + 1),
            BatchSize::LargeInput,
        )
    });
    g.finish();
}

/// The visited-set operations of the flip graph: hashing and comparing
/// triangulations.
fn hashing(c: &mut Criterion) {
    let mut g = c.benchmark_group("hashing");
    let chi = grid(3, 4).chirotope().unwrap();
    let all: Vec<Triangulation> = ExtensionGraph::new(&chi, false).unwrap().collect();
    g.bench_function("hashset_insert/grid_3x4", |b| {
        b.iter(|| all.iter().cloned().collect::<HashSet<_>>().len())
    });
    let set: HashSet<Triangulation> = all.iter().cloned().collect();
    g.bench_function("hashset_lookup/grid_3x4", |b| {
        b.iter(|| all.iter().filter(|t| set.contains(*t)).count())
    });
    g.finish();
}

criterion_group!(benches, construction, flips, flip_graph, extension, hashing);
criterion_main!(benches);
