use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ebi_arithmetic::fraction::fraction_exact::FractionExact;
use ebi_arithmetic::fraction::fraction_f64::FractionF64;
use ebi_arithmetic::matrix::fraction_matrix_exact::FractionMatrixExact;
use ebi_arithmetic::matrix::fraction_matrix_f64::FractionMatrixF64;
use ebi_arithmetic::matrix::mul_gpu::{
    run_mul_approx_f32, run_mul_approx_f64, run_mul_exact_signed_u64,
};
use ebi_arithmetic::shader::state::COMPUTE_SHADERS;
use itertools::izip;
use rand::Rng;
use std::hint::black_box;

pub fn bench_matrix_gpu_approx(c: &mut Criterion) {
    let mut group = c.benchmark_group("Matrix GPU Approx");
    group.sample_size(10);

    // Values divisible by 16 for optimal GPU performance
    for size in [
        64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152,
        1216, 1280, 1344, 1408, 1472,
    ]
    .iter()
    {
        let mut rng = rand::rng();
        let numerators = vec![rng.random_range(1..5); size * size];
        let denominators = vec![rng.random_range(6..10); size * size];

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
            b.iter(|| black_box(black_box(&matrix_f64) * black_box(&matrix_f64)).unwrap())
        });

        group.bench_function(BenchmarkId::new("Approx GPU f32", size), |b| {
            b.iter(|| {
                run_mul_approx_f32(
                    &numerators,
                    &denominators,
                    *size,
                    COMPUTE_SHADERS.get_matrix_mul_shader_f32(),
                    8,
                )
            })
        });

        if COMPUTE_SHADERS.get_matrix_mul_shader_f64().is_available() {
            group.bench_function(BenchmarkId::new("Approx GPU f64", size), |b| {
                b.iter(|| {
                    run_mul_approx_f64(
                        &numerators,
                        &denominators,
                        *size,
                        COMPUTE_SHADERS.get_matrix_mul_shader_f64(),
                        4,
                    )
                })
            });
        }
    }
}

pub fn bench_matrix_gpu_exact(c: &mut Criterion) {
    let mut group = c.benchmark_group("Matrix GPU Exact");
    group.sample_size(10);

    for size in [
        64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, /*832, 896, 960, 1024, 1088, 1152,
        1216, 1280, 1344, 1408, 1472,*/
    ]
    .iter()
    {
        let mut rng = rand::rng();
        let numerators = vec![rng.random_range(1..5); size * size];
        let denominators = vec![rng.random_range(6..10); size * size];

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
            b.iter(|| black_box(black_box(&matrix_exact) * black_box(&matrix_exact)).unwrap())
        });

        group.bench_function(BenchmarkId::new("Exact GPU Signed u64", size), |b| {
            b.iter(|| run_mul_exact_signed_u64(&numerators, &denominators, *size, 8u32))
        });

        /*group.bench_function(BenchmarkId::new("Exact GPU i64", size), |b| {
            b.iter(|| run_mul_exact_i64(&numerators, &denominators, *size, 8u32))
        });*/

        /*group.bench_function(BenchmarkId::new("Exact GPU i128", size), |b| {
            b.iter(|| run_mul_exact_i128(numerators.clone(), denominators.clone(), *size))
        });*/
    }
}

pub fn bench_matrix_gpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("Matrix GPU");
    group.sample_size(100);

    for size in [
        64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152,
        1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 2048,
    ]
    .iter()
    {
        let mut rng = rand::rng();
        let numerators = vec![rng.random_range(1..5); size * size];
        let denominators = vec![rng.random_range(6..10); size * size];

        let matrix_f64 = FractionMatrixF64 {
            number_of_columns: *size,
            number_of_rows: *size,
            values: izip!(numerators.iter(), denominators.iter())
                .map(|(nom, den)| FractionF64::from((*nom as i64, *den)).0)
                .collect::<Vec<_>>(),
        };

        let matrix_exact = FractionMatrixExact {
            number_of_columns: *size,
            number_of_rows: *size,
            values: izip!(numerators.iter(), denominators.iter())
                .map(|(nom, den)| FractionExact::from((*nom as i64, *den)).0)
                .collect::<Vec<_>>(),
        };

        // Init Shaders
        let _ = COMPUTE_SHADERS.get_matrix_mul_shader_exact_u64();

        /*group.bench_function(BenchmarkId::new("Exact CPU", size), |b| {
            b.iter(|| (&matrix_exact * &matrix_exact).unwrap())
        });*/

        group.bench_function(BenchmarkId::new("Exact GPU Signed U64", size), |b| {
            b.iter(|| run_mul_exact_signed_u64(&numerators, &denominators, *size, 8u32))
        });

        /*group.bench_function(BenchmarkId::new("Approx CPU", size), |b| {
            b.iter(|| (&matrix_f64 * &matrix_f64).unwrap())
        });*/

        group.bench_function(BenchmarkId::new("Approx GPU f32", size), |b| {
            b.iter(|| {
                run_mul_approx_f32(
                    &numerators,
                    &denominators,
                    *size,
                    COMPUTE_SHADERS.get_matrix_mul_shader_f32(),
                    8,
                )
            })
        });

        if COMPUTE_SHADERS.get_matrix_mul_shader_f64().is_available() {
            group.bench_function(BenchmarkId::new("Approx GPU f64", size), |b| {
                b.iter(|| {
                    run_mul_approx_f64(
                        &numerators,
                        &denominators,
                        *size,
                        COMPUTE_SHADERS.get_matrix_mul_shader_f64(),
                        8,
                    )
                })
            });
        }
    }
}

criterion_group!(
    matrix_gpu,
    bench_matrix_gpu_approx,
    bench_matrix_gpu_exact,
    bench_matrix_gpu
);
criterion_main!(matrix_gpu);
