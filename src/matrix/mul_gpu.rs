use crate::fraction::signed::Numerator;
use crate::matrix::fraction_matrix_exact::FractionMatrixExact;
use crate::matrix::fraction_matrix_f64::FractionMatrixF64;
use crate::shader::matrix_mul::{
    Dimensions, GpuRationalU32, GpuRationalU64, GpuSignedU64, MatrixMulShader,
};
use crate::shader::state::COMPUTE_SHADERS;
use crate::{EbiMatrix, One, Signed, Zero};
use itertools::izip;
use malachite::base::num::arithmetic::traits::Lcm;
use malachite::base::num::arithmetic::traits::UnsignedAbs;
use malachite::rational::Rational;
use malachite::{Integer, Natural};
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;

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

        let u64support = COMPUTE_SHADERS
            .get_matrix_mul_shader_exact_u64()
            .is_available();
        let bound = if u64support {
            Natural::from(u64::MAX)
        } else {
            Natural::from(u32::MAX)
        };

        if numerators
            .iter()
            .chain(numerators2.iter())
            .any(|i| i > &bound)
            || denominators
                .iter()
                .chain(denominators2.iter())
                .any(|n| n > &bound)
        {
            // Matrix entries do not fit into supported type
            return None;
        }

        // Try method: Direct multiplication on the GPU, if no overflows are guaranteed.
        if u64support {
            let rationals = izip!(numerators.iter(), denominators.iter())
                .map(|(n, d)| GpuRationalU64 {
                    sign: if n.is_negative() { 0 } else { 1 },
                    num: n.unsigned_abs().limbs()[0] as u64,
                    den: d.limbs()[0] as u64,
                })
                .collect::<Vec<_>>();
            let rationals2 = izip!(numerators2.iter(), denominators2.iter())
                .map(|(n, d)| GpuRationalU64 {
                    sign: if n.is_negative() { 0 } else { 1 },
                    num: n.unsigned_abs().limbs()[0] as u64,
                    den: d.limbs()[0] as u64,
                })
                .collect::<Vec<_>>();

            let new = COMPUTE_SHADERS.get_matrix_mul_shader_exact_u64().execute(
                rationals,
                rationals2,
                Dimensions {
                    n: n as u32,
                    m: m as u32,
                    p: p as u32,
                },
                1,
            );

            let mut values = vec![Rational::zero(); new.len()];
            for (idx, val) in new.iter().enumerate() {
                if val.den == 0 {
                    // Overflow
                    return None;
                }
                values[idx] = Rational::from(if val.sign == 1 { 1 } else { -1 })
                    * Rational::from(val.num)
                    / Rational::from(val.den);
            }

            Some(FractionMatrixExact {
                number_of_columns: n,
                number_of_rows: p,
                values,
            })
        } else {
            let rationals = izip!(numerators.iter(), denominators.iter())
                .map(|(n, d)| GpuRationalU32 {
                    sign: if n.is_negative() { 0 } else { 1 },
                    num: n.unsigned_abs().limbs()[0] as u32,
                    den: d.limbs()[0] as u32,
                })
                .collect::<Vec<_>>();
            let rationals2 = izip!(numerators2.iter(), denominators2.iter())
                .map(|(n, d)| GpuRationalU32 {
                    sign: if n.is_negative() { 0 } else { 1 },
                    num: n.unsigned_abs().limbs()[0] as u32,
                    den: d.limbs()[0] as u32,
                })
                .collect::<Vec<_>>();

            let new = COMPUTE_SHADERS.get_matrix_mul_shader_exact_u32().execute(
                rationals,
                rationals2,
                Dimensions {
                    n: n as u32,
                    m: m as u32,
                    p: p as u32,
                },
                1,
            );

            let mut values = vec![Rational::zero(); new.len()];
            for (idx, val) in new.iter().enumerate() {
                if val.den == 0 {
                    // Overflow
                    return None;
                }
                values[idx] = Rational::from(if val.sign == 1 { 1 } else { -1 })
                    * Rational::from(val.num)
                    / Rational::from(val.den);
            }

            Some(FractionMatrixExact {
                number_of_columns: n,
                number_of_rows: p,
                values,
            })
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
                    value: v.unsigned_abs_ref().limbs()[0] as u64,
                    sign: if v.is_negative() { 0 } else { 1 },
                })
                .collect::<Vec<_>>();
            let scaled_signed2 = scaled2
                .into_par_iter()
                .map(|v| GpuSignedU64 {
                    value: v.unsigned_abs_ref().limbs()[0] as u64,
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
                1,
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
                                    * Rational::from(if val.sign == 0 { 1 } else { -1 })
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
                1,
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
                    1,
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

pub fn run_mul_approx_f32(
    numerators: &Vec<u64>,
    denominators: &Vec<u64>,
    size: usize,
    shader: &MatrixMulShader<f32>,
    tiling: u32,
) {
    let values: Vec<f32> = izip!(numerators.iter(), denominators.iter())
        .map(|(num, denom)| (*num as f32) / (*denom as f32))
        .collect::<Vec<_>>();
    shader.execute(
        values.clone(),
        values.clone(),
        Dimensions {
            n: size as u32,
            m: size as u32,
            p: size as u32,
        },
        tiling,
    );
}

pub fn run_mul_approx_f64(
    numerators: &Vec<u64>,
    denominators: &Vec<u64>,
    size: usize,
    shader: &MatrixMulShader<f64>,
    tiling: u32,
) {
    let values: Vec<f64> = izip!(numerators.iter(), denominators.iter())
        .map(|(num, denom)| (*num as f64) / (*denom as f64))
        .collect::<Vec<_>>();
    shader.execute(
        values.clone(),
        values.clone(),
        Dimensions {
            n: size as u32,
            m: size as u32,
            p: size as u32,
        },
        tiling,
    );
}

pub fn run_mul_exact(numerators: &Vec<u64>, denominators: &Vec<u64>, size: usize) {
    let rationals = izip!(numerators.iter(), denominators.iter())
        .map(|(n, d)| GpuRationalU32 {
            sign: if n.is_negative() { 0 } else { 1 },
            num: *n as u32,
            den: *d as u32,
        })
        .collect::<Vec<_>>();
    let rationals2 = izip!(numerators.iter(), denominators.iter())
        .map(|(n, d)| GpuRationalU32 {
            sign: if n.is_negative() { 0 } else { 1 },
            num: *n as u32,
            den: *d as u32,
        })
        .collect::<Vec<_>>();

    COMPUTE_SHADERS.get_matrix_mul_shader_exact_u32().execute(
        rationals,
        rationals2,
        Dimensions {
            n: size as u32,
            m: size as u32,
            p: size as u32,
        },
        1,
    );
}

pub fn run_mul_exact_i64(numerators: &Vec<u64>, denominators: &Vec<u64>, size: usize) {
    let row_lcms: Vec<Natural> = (0..size)
        .into_par_iter()
        .map(|i| {
            (0..size).fold(Natural::one(), |acc, j| {
                acc.lcm(Natural::from(denominators[i * size + j]))
            })
        })
        .collect();
    let col_lcms: Vec<Natural> = (0..size)
        .into_par_iter()
        .map(|j| {
            (0..size).fold(Natural::one(), |acc, i| {
                acc.lcm(Natural::from(denominators[i * size + j]))
            })
        })
        .collect();
    let scaled: Vec<Integer> = (0..size)
        .into_par_iter()
        .flat_map(|i| {
            (0..size)
                .map(|j| {
                    let index = i * size + j;
                    let factor = &row_lcms[i] / Natural::from(denominators[index]);
                    Integer::from(numerators[index]) * Integer::from(factor)
                })
                .collect::<Vec<_>>()
        })
        .collect();
    let scaled2: Vec<Integer> = (0..size)
        .into_par_iter()
        .flat_map(|i| {
            (0..size)
                .map(|j| {
                    let index = i * size + j;
                    let factor = &col_lcms[j] / Natural::from(denominators[index]);
                    Integer::from(numerators[index]) * Integer::from(factor)
                })
                .collect::<Vec<_>>()
        })
        .collect();
    let scaled_signed = scaled
        .into_par_iter()
        .map(|v| GpuSignedU64 {
            value: v.unsigned_abs_ref().limbs()[0] as u64,
            sign: if v.is_negative() { 0 } else { 1 },
        })
        .collect::<Vec<_>>();
    let scaled_signed2 = scaled2
        .into_par_iter()
        .map(|v| GpuSignedU64 {
            value: v.unsigned_abs_ref().limbs()[0] as u64,
            sign: if v.is_negative() { 0 } else { 1 },
        })
        .collect::<Vec<_>>();

    let new_scaled = COMPUTE_SHADERS.get_matrix_mul_shader_signed_u64().execute(
        scaled_signed,
        scaled_signed2,
        Dimensions {
            n: size as u32,
            m: size as u32,
            p: size as u32,
        },
        1,
    );
    Some(FractionMatrixExact {
        number_of_columns: size,
        number_of_rows: size,
        values: (0..size)
            .into_par_iter()
            .flat_map(|i| {
                (0..size)
                    .map(|j| {
                        let idx = i * size + j;
                        let val = &new_scaled[idx];
                        Rational::from(val.value)
                            * Rational::from(if val.sign == 0 { 1 } else { -1 })
                            / (Rational::from(&row_lcms[i] * &col_lcms[j]))
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>(),
    });
}

pub fn run_mul_exact_i128(numerators: &Vec<u64>, denominators: &Vec<u64>, size: usize) {
    /*let row_lcms: Vec<Natural> = (0..size)
        .into_par_iter()
        .map(|i| {
            (0..size).fold(Natural::one(), |acc, j| {
                acc.lcm(Natural::from(denominators[i * size + j]))
            })
        })
        .collect();
    let col_lcms: Vec<Natural> = (0..size)
        .into_par_iter()
        .map(|j| {
            (0..size).fold(Natural::one(), |acc, i| {
                acc.lcm(Natural::from(denominators[i * size + j]))
            })
        })
        .collect();
    let scaled: Vec<Integer> = (0..size)
        .into_par_iter()
        .flat_map(|i| {
            (0..size)
                .map(|j| {
                    let index = i * size + j;
                    let factor = &row_lcms[i] / Natural::from(denominators[index]);
                    Integer::from(numerators[index]) * Integer::from(factor)
                })
                .collect::<Vec<_>>()
        })
        .collect();
    let scaled2: Vec<Integer> = (0..size)
        .into_par_iter()
        .flat_map(|i| {
            (0..size)
                .map(|j| {
                    let index = i * size + j;
                    let factor = &col_lcms[j] / Natural::from(denominators[index]);
                    Integer::from(numerators[index]) * Integer::from(factor)
                })
                .collect::<Vec<_>>()
        })
        .collect();
    let scaled_signed = scaled
        .into_par_iter()
        .map(|v| {
            let limbs = v
                .unsigned_abs_ref()
                .to_limbs_asc()
                .iter()
                .map(|x| *x)
                .collect::<Vec<_>>();
            GpuI128 {
                number: [
                    *limbs.get(3).unwrap_or(&0u32),
                    *limbs.get(2).unwrap_or(&0u32),
                    *limbs.get(1).unwrap_or(&0u32),
                    *limbs.get(0).unwrap_or(&0u32),
                ],
                sign: if v.is_negative() { 0 } else { 1 },
            }
        })
        .collect::<Vec<_>>();
    let scaled_signed2 = scaled2
        .into_par_iter()
        .map(|v| {
            let limbs = v
                .unsigned_abs_ref()
                .to_limbs_asc()
                .iter()
                .map(|x| *x)
                .collect::<Vec<_>>();
            GpuI128 {
                number: [
                    *limbs.get(3).unwrap_or(&0u32),
                    *limbs.get(2).unwrap_or(&0u32),
                    *limbs.get(1).unwrap_or(&0u32),
                    *limbs.get(0).unwrap_or(&0u32),
                ],
                sign: if v.is_negative() { 0 } else { 1 },
            }
        })
        .collect::<Vec<_>>();

    COMPUTE_SHADERS.get_matrix_mul_shader_i128().execute(
        scaled_signed,
        scaled_signed2,
        Dimensions {
            n: size as u32,
            m: size as u32,
            p: size as u32,
        },
    );*/
}

#[cfg(test)]
mod tests {
    use crate::EbiMatrix;
    use crate::fraction::fraction_exact::FractionExact;
    use crate::matrix::fraction_matrix_exact::FractionMatrixExact;
    use crate::matrix::mul_gpu::MulGpu;
    use itertools::izip;
    use rand::Rng;

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

        //let prod = (&m1).mul_gpu(&m2).unwrap();
        let prod_transformed = (&m1).mul_gpu_transformed(&m2).unwrap();

        //assert_eq!(prod.number_of_columns(), 2);
        //assert_eq!(prod.number_of_rows(), 2);

        let m3 = vec![
            vec![FractionExact::from(-8), FractionExact::from(24)],
            vec![FractionExact::from(-83), FractionExact::from(154)],
        ];

        //assert_eq!(prod.clone().to_vec(), m3);
        assert_eq!(prod_transformed.clone().to_vec(), m3);
    }

    #[test]
    fn fraction_matrix_exact_mul_random() {
        let size = 200;
        let mut rng = rand::rng();
        let sqrt = 10_u64;
        let numerators: Vec<u64> = (0..size * size)
            .map(|_| rng.random_range(0..sqrt))
            .collect();
        let denominators: Vec<u64> = (0..size * size)
            .map(|_| rng.random_range(1..sqrt))
            .collect();

        let matrix1: FractionMatrixExact = FractionMatrixExact {
            number_of_columns: size,
            number_of_rows: size,
            values: izip!(numerators.iter(), denominators.iter())
                .map(|(nom, den)| FractionExact::from((*nom as i64, *den)).0)
                .collect::<Vec<_>>(),
        };

        let matrix2: FractionMatrixExact = FractionMatrixExact {
            number_of_columns: size,
            number_of_rows: size,
            values: izip!(numerators.iter(), denominators.iter())
                .map(|(nom, den)| FractionExact::from((*nom as i64, *den)).0)
                .collect::<Vec<_>>(),
        };

        let res_ref = (&matrix1 * &matrix2).unwrap();
        let res = (&matrix1).mul_gpu_transformed(&matrix2).unwrap();

        assert_eq!(res_ref.clone().to_vec(), res.clone().to_vec());
    }
}
