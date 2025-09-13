use crate::fraction::signed::Numerator;
use crate::matrix::fraction_matrix_exact::FractionMatrixExact;
use crate::matrix::fraction_matrix_f64::FractionMatrixF64;
use crate::shader::matrix_mul::{Dimensions, GpuRationalU32, GpuRationalU64, GpuSignedU64};
use crate::shader::state::COMPUTE_SHADERS;
use crate::{EbiMatrix, One, Signed, Zero};
use itertools::izip;
use itertools::Itertools;
use malachite::base::num::arithmetic::traits::{Lcm, Sign, UnsignedAbs};
use malachite::rational::Rational;
use malachite::{Integer, Natural};
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator};

pub trait MulGpu {
    type Output;

    fn mul_gpu(self, rhs: Self) -> Self::Output;

    fn mul_gpu_transformed(self, rhs: Self) -> Self::Output;
}

impl MulGpu for &FractionMatrixExact {
    type Output = Option<FractionMatrixExact>;

    fn mul_gpu(self, rhs: Self) -> Self::Output {
        if !COMPUTE_SHADERS
            .get_matrix_mul_shader_exact_u32()
            .is_available()
            && !COMPUTE_SHADERS
                .get_matrix_mul_shader_exact_u64()
                .is_available()
        {
            return None;
        }

        let n = self.number_of_rows();
        let m = self.number_of_columns();
        let p = rhs.number_of_columns();

        let numerators = self.numerators();
        let numerators2 = rhs.numerators();
        let denominators = self.denominators();
        let denominators2 = self.denominators();

        // Try method: Direct multiplication on the GPU, if no overflows are guaranteed.
        // TODO: The current GPU impl can only support u32, find a way to support u64
        let num_bound = numerators
            .iter()
            .map(|val| val.unsigned_abs())
            .max()
            .unwrap_or(Natural::zero())
            * numerators2
                .iter()
                .map(|val| val.unsigned_abs())
                .max()
                .unwrap_or(Natural::zero())
            * Natural::from(m);
        let den_bound = denominators.iter().min().unwrap_or(&Natural::zero())
            * denominators2.iter().min().unwrap_or(&Natural::one());

        macro_rules! shader_exact {
            ($v:ident, $u:ident, $name:tt) => {
                if num_bound <= Natural::from($v::MAX) && den_bound <= Natural::from($v::MAX) {
                    let rationals = izip!(numerators.iter(), denominators.iter())
                        .map(|(n, d)| $u {
                            sign: if n.is_negative() { 0 } else { 1 },
                            num: n.unsigned_abs().limbs()[0] as $v,
                            den: d.limbs()[0] as $v,
                        })
                        .collect::<Vec<_>>();
                    let rationals2 = izip!(numerators2.iter(), denominators2.iter())
                        .map(|(n, d)| $u {
                            sign: if n.is_negative() { 0 } else { 1 },
                            num: n.unsigned_abs().limbs()[0] as $v,
                            den: d.limbs()[0] as $v,
                        })
                        .collect::<Vec<_>>();

                    let new = COMPUTE_SHADERS.$name().execute(
                        rationals,
                        rationals2,
                        Dimensions {
                            n: n as u32,
                            m: m as u32,
                            p: p as u32,
                        },
                    );

                    Some(FractionMatrixExact {
                        number_of_columns: n,
                        number_of_rows: p,
                        values: new
                            .iter()
                            .map(|x| {
                                Rational::from(if x.sign == 1 { 1 } else { -1 })
                                    * Rational::from(x.num)
                                    / Rational::from(x.den)
                            })
                            .collect::<Vec<_>>(),
                    })
                } else {
                    None
                }
            };
        }

        if COMPUTE_SHADERS
            .get_matrix_mul_shader_exact_u64()
            .is_available()
        {
            shader_exact!(u64, GpuRationalU64, get_matrix_mul_shader_exact_u64)
        } else {
            shader_exact!(u32, GpuRationalU32, get_matrix_mul_shader_exact_u32)
        }
    }

    fn mul_gpu_transformed(self, rhs: Self) -> Self::Output {
        if !COMPUTE_SHADERS
            .get_matrix_mul_shader_signed_u64()
            .is_available()
        {
            return None;
        }

        let n = self.number_of_rows();
        let m = self.number_of_columns();
        let p = rhs.number_of_columns();

        let row_lcms: Vec<Natural> = (0..n)
            .into_par_iter()
            .map(|i| {
                (0..m).fold(Natural::one(), |acc, j| {
                    acc.lcm(self.values[self.index(i, j)].denominator_ref())
                })
            })
            .collect();
        let col_lcms: Vec<Natural> = (0..p)
            .into_par_iter()
            .map(|j| {
                (0..m).fold(Natural::one(), |acc, i| {
                    acc.lcm(rhs.values[rhs.index(i, j)].denominator_ref())
                })
            })
            .collect();

        let scaled: Vec<Integer> = (0..n)
            .into_par_iter()
            .flat_map(|i| {
                (0..m)
                    .map(|j| {
                        let val = &self.values[self.index(i, j)];
                        let factor = &row_lcms[i] / val.denominator_ref();
                        val.signed_numerator() * Integer::from(factor)
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        let scaled2: Vec<Integer> = (0..m)
            .into_par_iter()
            .flat_map(|i| {
                (0..p)
                    .map(|j| {
                        let val = &rhs.values[rhs.index(i, j)];
                        let factor = &col_lcms[j] / val.denominator_ref();
                        val.signed_numerator() * Integer::from(factor)
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        let num_bound = scaled
            .iter()
            .map(|x| x.unsigned_abs())
            .max()
            .unwrap_or(Natural::zero())
            * scaled2
                .iter()
                .map(|x| x.unsigned_abs())
                .max()
                .unwrap_or(Natural::zero())
            * Natural::from(m);

        if num_bound <= Integer::from(u64::MAX) {
            let scaled_signed = scaled
                .into_par_iter()
                .map(|v| GpuSignedU64 {
                    value: v.clone().unsigned_abs().limbs()[0] as u64,
                    sign: if v.is_negative() { 0 } else { 1 },
                })
                .collect::<Vec<_>>();
            let scaled_signed2 = scaled2
                .into_par_iter()
                .map(|v| GpuSignedU64 {
                    value: v.clone().unsigned_abs().limbs()[0] as u64,
                    sign: if v.is_negative() { 0 } else { 1 },
                })
                .collect::<Vec<_>>();

            let new_scaled = COMPUTE_SHADERS.get_matrix_mul_shader_signed_u64().execute(
                scaled_signed,
                scaled_signed2,
                Dimensions {
                    n: n as u32,
                    m: m as u32,
                    p: p as u32,
                },
            );

            Some(FractionMatrixExact {
                number_of_columns: n,
                number_of_rows: p,
                values: (0..n)
                    .into_par_iter()
                    .flat_map(|i| {
                        (0..p)
                            .map(|j| {
                                let idx = i * p + j;
                                let val = &new_scaled[idx];
                                Rational::from(val.value)
                                    * Rational::from(if val.sign == 1 { -1 } else { 1 })
                                    / (Rational::from(&row_lcms[i] * &col_lcms[j]))
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>(),
            })
        } else {
            None
        }
    }
}

impl MulGpu for &FractionMatrixF64 {
    type Output = Option<FractionMatrixF64>;

    fn mul_gpu(self, rhs: Self) -> Self::Output {
        if !COMPUTE_SHADERS.get_matrix_mul_shader_f32().is_available()
            && !COMPUTE_SHADERS.get_matrix_mul_shader_f64().is_available()
        {
            return None;
        }

        let n = self.number_of_rows();
        let m = self.number_of_columns();
        let p = rhs.number_of_columns();

        let values = if COMPUTE_SHADERS.get_matrix_mul_shader_f64().is_available() {
            COMPUTE_SHADERS.get_matrix_mul_shader_f64().execute(
                self.values.clone(),
                rhs.values.clone(),
                Dimensions {
                    n: n as u32,
                    m: m as u32,
                    p: p as u32,
                },
            )
        } else {
            // TODO: Only works with f32, so less precision. How to find out that less precision is sufficient?
            COMPUTE_SHADERS
                .get_matrix_mul_shader_f32()
                .execute(
                    self.values
                        .iter()
                        .map(|val| *val as f32)
                        .collect::<Vec<f32>>(),
                    rhs.values
                        .iter()
                        .map(|val| *val as f32)
                        .collect::<Vec<f32>>(),
                    Dimensions {
                        n: n as u32,
                        m: m as u32,
                        p: p as u32,
                    },
                )
                .iter()
                .map(|val| *val as f64)
                .collect::<Vec<f64>>()
        };

        Some(FractionMatrixF64 {
            values,
            number_of_columns: n,
            number_of_rows: p,
        })
    }

    fn mul_gpu_transformed(self, _: Self) -> Self::Output {
        None
    }
}

#[cfg(test)]
mod tests {
    use crate::exact::MaybeExact;
    use crate::fraction::fraction_exact::FractionExact;
    use crate::fraction::fraction_f64::FractionF64;
    use crate::matrix::fraction_matrix_exact::FractionMatrixExact;
    use crate::matrix::fraction_matrix_f64::FractionMatrixF64;
    use crate::matrix::mul_gpu::MulGpu;
    use crate::shader::state::COMPUTE_SHADERS;
    use crate::EbiMatrix;
    use itertools::izip;
    use rand::Rng;
    use std::time::Instant;

    #[test]
    fn fraction_matrix_exact_mul() {
        let m1: FractionMatrixExact = vec![
            vec![
                FractionExact::from(1),
                FractionExact::from(2),
                FractionExact::from(3),
            ],
            vec![
                FractionExact::from(4),
                FractionExact::from(-5),
                FractionExact::from(6),
            ],
        ]
        .try_into()
        .unwrap();

        let m2: FractionMatrixExact = vec![
            vec![FractionExact::from(7), FractionExact::from(8)],
            vec![FractionExact::from(9), FractionExact::from(-10)],
            vec![FractionExact::from(-11), FractionExact::from(12)],
        ]
        .try_into()
        .unwrap();

        let prod = (&m1).mul_gpu(&m2).unwrap();
        let prod_transformed = (&m1).mul_gpu_transformed(&m2).unwrap();

        assert_eq!(prod.number_of_columns(), 2);
        assert_eq!(prod.number_of_rows(), 2);

        let m3 = vec![
            vec![FractionExact::from(-8), FractionExact::from(24)],
            vec![FractionExact::from(-83), FractionExact::from(154)],
        ];

        assert_eq!(prod.clone().to_vec(), m3);
        assert_eq!(prod_transformed.clone().to_vec(), m3);
    }

    #[test]
    fn bench_mul_gpu() {
        let repeat = 1;
        let size = 500_usize;

        let mut rng = rand::thread_rng();
        let sqrt = 10000_u64;
        let numerators = vec![rng.gen_range(0..sqrt); size * size];
        let denominators = vec![rng.gen_range(0..sqrt); size * size];

        let matrices_f64: Vec<FractionMatrixF64> = (0..repeat)
            .into_iter()
            .map(|i| FractionMatrixF64 {
                number_of_columns: size,
                number_of_rows: size,
                values: numerators
                    .iter()
                    .zip(denominators.iter())
                    .enumerate()
                    .map(|(x, (nom, den))| {
                        if x == i {
                            FractionF64::from((*nom as i64, den + 1)).0
                        } else {
                            FractionF64::from((*nom as i64, *den)).0
                        }
                    })
                    .collect::<Vec<_>>(),
            })
            .collect();

        let matrices_exact: Vec<FractionMatrixExact> = (0..repeat)
            .into_iter()
            .map(|i| FractionMatrixExact {
                number_of_columns: size,
                number_of_rows: size,
                values: numerators
                    .iter()
                    .zip(denominators.iter())
                    .enumerate()
                    .map(|(x, (nom, den))| {
                        if x == i {
                            FractionExact::from((*nom as i64, den + 1)).0
                        } else {
                            FractionExact::from((*nom as i64, *den)).0
                        }
                    })
                    .collect(),
            })
            .collect();

        // Init shaders
        {
            let before = Instant::now();
            let _ = COMPUTE_SHADERS.get_matrix_mul_shader_f32();
            println!("init shaders:      {:.2?}", before.elapsed());
        }

        let mut matrices_exact_results: Vec<FractionMatrixExact> = vec![];

        // exact u64 cpu
        {
            let before = Instant::now();
            for m in &matrices_exact {
                let m3 = (m * m).unwrap();

                if !m3.is_exact() {
                    panic!()
                }
                matrices_exact_results.push(m3)
            }

            println!("exact u64 cpu:     {:.2?}", before.elapsed());
        }

        // exact u64 gpu
        /*{
            let before = Instant::now();
            for (m, res) in izip!(matrices_exact.iter(), matrices_exact_results.iter()) {
                let m3 = m.mul_gpu(m).unwrap();

                if !m3.is_exact() {
                    panic!()
                }
                assert_eq!(res.clone().to_vec(), m3.to_vec());
            }

            println!("exact u64 gpu:     {:.2?}", before.elapsed());
        }*/

        // exact u64 gpu transformed
        {
            let before = Instant::now();
            for (m, res) in izip!(matrices_exact.iter(), matrices_exact_results.iter()) {
                let m3 = m.mul_gpu_transformed(m).unwrap();

                if !m3.is_exact() {
                    panic!()
                }
                assert_eq!(res.clone().to_vec(), m3.to_vec());
            }

            println!("exact u64 gpu transformed:     {:.2?}", before.elapsed());
        }

        // f64 gpu
        {
            let before = Instant::now();
            for m in &matrices_f64 {
                let m3 = m.mul_gpu(m).unwrap();

                if m3.is_exact() {
                    panic!()
                }
            }

            println!("approx f64 gpu:    {:.2?}", before.elapsed());
        }

        // f64 cpu
        {
            let before = Instant::now();
            for m in &matrices_f64 {
                let m3 = (m * m).unwrap();

                if m3.is_exact() {
                    panic!()
                }
            }

            println!("approx f64 cpu:    {:.2?}", before.elapsed());
        }
    }
}
