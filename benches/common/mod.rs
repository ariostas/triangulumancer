//! Workloads shared by the benchmarks.
//!
//! The configurations are seeded and self-contained, and
//! `benches/compare_oracle.py` builds exactly the same ones (the same
//! SplitMix64 stream), so the Criterion numbers and the Python comparison
//! against the 0.3 oracle measure the same inputs.

#![allow(dead_code)]

use triangulumancer::config::{PointConfig, VectorConfig};

/// SplitMix64, as in the test harnesses; mirrored in `compare_oracle.py`.
pub struct Rng(pub u64);

impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in `0..bound` (modulo bias is irrelevant here).
    pub fn below(&mut self, bound: u64) -> i64 {
        (self.next_u64() % bound) as i64
    }

    /// Uniform in `[0, 1)`, from the top 53 bits.
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// `n` points with coordinates uniform in `0..range`, row-major.
pub fn random_coords(seed: u64, n: usize, dim: usize, range: u64) -> Vec<i64> {
    let mut rng = Rng(seed);
    (0..n * dim).map(|_| rng.below(range)).collect()
}

pub fn random_points(seed: u64, n: usize, dim: usize, range: u64) -> PointConfig {
    PointConfig::from_row_major(&random_coords(seed, n, dim, range), n, dim).unwrap()
}

/// Weights uniform in `[0, scale)`, drawn after the coordinates of the same
/// seed would have been.
pub fn random_weights(seed: u64, n: usize, scale: f64) -> Vec<f64> {
    let mut rng = Rng(seed ^ 0x5eed);
    (0..n).map(|_| rng.unit() * scale).collect()
}

/// The `a × b` lattice grid `{0..a} × {0..b}`, `x` major.
pub fn grid(a: i64, b: i64) -> PointConfig {
    let pts: Vec<i64> = (0..a)
        .flat_map(|x| (0..b).flat_map(move |y| [x, y]))
        .collect();
    PointConfig::from_row_major(&pts, (a * b) as usize, 2).unwrap()
}

/// The 16 vectors of rank 4 from 0.3's `test_vc_triangulation_with_degenerate_flip`
/// (`vc_degenerate_flip` in the fixture corpus): billions of triangulations.
pub fn vc_degenerate_flip() -> VectorConfig {
    #[rustfmt::skip]
    let rows = [
        3, 1, -2, -1,  0, 0, 0, 1,  -1, -3, 1, 1,  0, 2, 1, -1,
        2, 0, -1, 0,   1, -1, 0, 0, 1, 1, 0, 0,    0, 0, 1, 0,
        -1, 0, 0, 0,   1, 0, 0, 0,  0, -2, 1, 1,   3, 1, -1, -1,
        4, 3, -1, -2,  1, -1, 0, 1, 0, -3, 1, 2,   2, 2, 0, -1,
    ];
    VectorConfig::from_row_major(&rows, 16, 4).unwrap()
}
