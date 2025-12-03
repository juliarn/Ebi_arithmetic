use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ebi_arithmetic::fraction::fraction_exact::FractionExact;
use ebi_arithmetic::fraction::fraction_f64::FractionF64;
use ebi_arithmetic::matrix::fraction_matrix_exact::FractionMatrixExact;
use ebi_arithmetic::matrix::fraction_matrix_f64::FractionMatrixF64;
use ebi_arithmetic::matrix::mul_gpu::{
    run_mul_approx_f32, run_mul_approx_f64, run_mul_exact, run_mul_exact_i64,
};
use ebi_arithmetic::shader::state::COMPUTE_SHADERS;
use itertools::izip;
use rand::Rng;

pub fn bench_matrix_gpu_approx(c: &mut Criterion) {
    let mut group = c.benchmark_group("Matrix GPU Approx");

    for size in [
        10usize, 25, 50, 75, 100, 125, 150, 175, 200, /*225, 250, 275, 300, 325, 350, 375, 400, 425,
        450, 475, 500,*/
    ]
    .iter()
    {
        let mut rng = rand::rng();
        let sqrt = 100000_u64;
        let numerators = vec![rng.random_range(0..sqrt); size * size];
        let denominators = vec![rng.random_range(1..sqrt); size * size];

        let matrix_f64 = FractionMatrixF64 {
            number_of_columns: *size,
            number_of_rows: *size,
            values: izip!(numerators.iter(), denominators.iter())
                .map(|(nom, den)| FractionF64::from((*nom as i64, *den)).0)
                .collect::<Vec<_>>(),
        };

        // Init Shaders
        let _ = COMPUTE_SHADERS.get_matrix_mul_shader_f32();

        group.bench_function(BenchmarkId::new("Approx CPU", size), |b| {
            b.iter(|| (&matrix_f64 * &matrix_f64).unwrap())
        });

        group.bench_function(BenchmarkId::new("Approx GPU f32", size), |b| {
            b.iter(|| run_mul_approx_f32(numerators.clone(), denominators.clone(), *size))
        });

        if COMPUTE_SHADERS.get_matrix_mul_shader_f64().is_available() {
            group.bench_function(BenchmarkId::new("Approx GPU f64", size), |b| {
                b.iter(|| run_mul_approx_f64(numerators.clone(), denominators.clone(), *size))
            });
        }
    }
}

pub fn bench_matrix_gpu_exact(c: &mut Criterion) {
    let mut group = c.benchmark_group("Matrix GPU Exact");

    for size in [10usize, 50, 100, 150, 200].iter() {
        let mut rng = rand::rng();
        let sqrt = 100000_u64;
        let numerators = vec![rng.random_range(0..sqrt); size * size];
        let denominators = vec![rng.random_range(1..sqrt); size * size];

        let matrix_exact = FractionMatrixExact {
            number_of_columns: *size,
            number_of_rows: *size,
            values: izip!(numerators.iter(), denominators.iter())
                .map(|(nom, den)| FractionExact::from((*nom as i64, *den)).0)
                .collect::<Vec<_>>(),
        };

        // Init Shaders
        let _ = COMPUTE_SHADERS.get_matrix_mul_shader_exact_u64();

        group.bench_function(BenchmarkId::new("Exact CPU", size), |b| {
            b.iter(|| (&matrix_exact * &matrix_exact).unwrap())
        });

        group.bench_function(BenchmarkId::new("Exact GPU", size), |b| {
            b.iter(|| run_mul_exact(numerators.clone(), denominators.clone(), *size))
        });

        group.bench_function(BenchmarkId::new("Exact GPU i64", size), |b| {
            b.iter(|| run_mul_exact_i64(numerators.clone(), denominators.clone(), *size))
        });

        /*group.bench_function(BenchmarkId::new("Exact GPU i128", size), |b| {
            b.iter(|| run_mul_exact_i128(numerators.clone(), denominators.clone(), *size))
        });*/
    }
}

criterion_group!(matrix_gpu, bench_matrix_gpu_approx, bench_matrix_gpu_exact);
criterion_main!(matrix_gpu);
