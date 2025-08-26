use crate::matrix::ebi_matrix::EbiMatrix;
use crate::matrix::fraction_matrix_exact::FractionMatrixExact;
use crate::matrix::fraction_matrix_f64::FractionMatrixF64;
use crate::matrix::loose_fraction::Type;
use crate::shader::matrix_mul::{Dimensions, Rational};
use crate::shader::state::COMPUTE_SHADERS;
use fraction::Integer;
use itertools::izip;
use num_bigint::{BigUint, ToBigUint};
use num_traits::{One, ToPrimitive, Zero};

pub trait MulGpu {
    type Output;

    fn mul_gpu(self, rhs: Self) -> Self::Output;

    fn mul_gpu_transformed(self, rhs: Self) -> Self::Output;
}

impl MulGpu for &FractionMatrixExact {
    type Output = Option<FractionMatrixExact>;

    fn mul_gpu(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (
                FractionMatrixExact::U64 {
                    types,
                    numerators,
                    denominators,
                    ..
                },
                FractionMatrixExact::U64 {
                    types: types2,
                    numerators: numerators2,
                    denominators: denominators2,
                    ..
                },
            ) => {
                let n = self.number_of_rows();
                let m = self.number_of_columns();
                let p = rhs.number_of_columns();

                // Try method: Direct multiplication on the GPU, if no overflows are guaranteed.
                // TODO: The current GPU impl can only support u32, find a way to support u64
                let num_bound = BigUint::from(*numerators.iter().max().unwrap_or(&0u64))
                    * BigUint::from(*numerators2.iter().max().unwrap_or(&0u64))
                    * BigUint::from(m);
                let den_bound = BigUint::from(*denominators.iter().min().unwrap_or(&1u64))
                    * BigUint::from(*denominators2.iter().min().unwrap_or(&1u64));

                println!("num_bound: {}", num_bound);
                println!("den_bound: {}", den_bound);

                if num_bound <= BigUint::from(u32::MAX) && den_bound <= BigUint::from(u32::MAX) {
                    let rationals = izip!(types.iter(), numerators.iter(), denominators.iter())
                        .map(|(t, n, d)| Rational {
                            sign: (t == &Type::Plus || t == &Type::Infinite || t == &Type::NaN)
                                as u32,
                            num: *n as u32,
                            den: *d as u32,
                        })
                        .collect::<Vec<Rational>>();
                    let rationals2 = izip!(types2.iter(), numerators2.iter(), denominators2.iter())
                        .map(|(t, n, d)| Rational {
                            sign: (t == &Type::Plus || t == &Type::Infinite || t == &Type::NaN)
                                as u32,
                            num: *n as u32,
                            den: *d as u32,
                        })
                        .collect::<Vec<Rational>>();

                    for r in &rationals {
                        println!("r: {:#?}", r);
                    }
                    println!("========");
                    for r in &rationals2 {
                        println!("r2: {:#?}", r);
                    }

                    let new = COMPUTE_SHADERS.matrix_mul_shader_exact().execute(
                        rationals,
                        rationals2,
                        Dimensions {
                            n: n as u32,
                            m: m as u32,
                            p: p as u32,
                        },
                    );

                    println!("Result:");
                    for r in &new {
                        println!("new: {:#?}", r);
                    }

                    Some(FractionMatrixExact::U64 {
                        number_of_columns: n,
                        number_of_rows: p,
                        types: new
                            .iter()
                            .map(|x| if x.sign == 1 { Type::Plus } else { Type::Minus })
                            .collect(),
                        numerators: new.iter().map(|x| x.num as u64).collect(),
                        denominators: new.iter().map(|x| x.den as u64).collect(),
                    })
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn mul_gpu_transformed(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (
                FractionMatrixExact::U64 {
                    types,
                    numerators,
                    denominators,
                    ..
                },
                FractionMatrixExact::U64 {
                    types: types2,
                    numerators: numerators2,
                    denominators: denominators2,
                    ..
                },
            ) => {
                let n = self.number_of_rows();
                let m = self.number_of_columns();
                let p = rhs.number_of_columns();

                // Try method: Pulling out the LCM of the denominators and scaling everything to
                // integer multiplication. Only perform if no overflows are guaranteed.
                let lcm = denominators
                    .iter()
                    .fold(BigUint::one(), |acc, denom| acc.lcm(&(*denom).into()));
                let lcm2 = denominators2
                    .iter()
                    .fold(BigUint::one(), |acc, denom| acc.lcm(&(*denom).into()));
                let scaled = numerators
                    .iter()
                    .zip(denominators.iter())
                    .map(|(num, den)| num * (&lcm / den))
                    .collect::<Vec<_>>();
                let scaled2 = numerators2
                    .iter()
                    .zip(denominators2.iter())
                    .map(|(num, den)| num * (&lcm2 / den))
                    .collect::<Vec<_>>();

                let num_bound = scaled.iter().max().unwrap_or(&BigUint::zero())
                    * scaled2.iter().max().unwrap_or(&BigUint::zero())
                    * BigUint::from(m);
                let denom = &lcm * &lcm2;

                println!("num_bound: {}", num_bound);
                println!("denom:     {}", denom);

                if num_bound <= i64::MAX.to_biguint().unwrap()
                    && denom <= i64::MAX.to_biguint().unwrap()
                {
                    let scaled_signed = scaled
                        .iter()
                        .zip(types.iter())
                        .map(|(x, typee)| {
                            x.to_i64().unwrap()
                                * if *typee == Type::Minus || *typee == Type::NegInfinite {
                                    -1
                                } else {
                                    1
                                }
                        })
                        .collect::<Vec<i64>>();
                    let scaled_signed2 = scaled2
                        .iter()
                        .zip(types2.iter())
                        .map(|(x, typee)| {
                            x.to_i64().unwrap()
                                * if *typee == Type::Minus || *typee == Type::NegInfinite {
                                    -1
                                } else {
                                    1
                                }
                        })
                        .collect::<Vec<i64>>();

                    let new_scaled = COMPUTE_SHADERS.matrix_mul_shader_u64().execute(
                        scaled_signed,
                        scaled_signed2,
                        Dimensions {
                            n: n as u32,
                            m: m as u32,
                            p: p as u32,
                        },
                    );

                    Some(FractionMatrixExact::U64 {
                        number_of_columns: n,
                        number_of_rows: p,
                        types: new_scaled
                            .iter()
                            .map(|x| if *x >= 0 { Type::Plus } else { Type::Minus })
                            .collect(),
                        numerators: new_scaled.iter().map(|x| x.abs() as u64).collect(),
                        denominators: vec![denom.to_u64().unwrap(); p * n],
                    })
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

impl MulGpu for &FractionMatrixF64 {
    type Output = Option<FractionMatrixF64>;

    fn mul_gpu(self, rhs: Self) -> Self::Output {
        let n = self.number_of_rows();
        let m = self.number_of_columns();
        let p = rhs.number_of_columns();

        // TODO: Only works with f32, so less precision. How to find out that less precision is sufficient?
        let values = COMPUTE_SHADERS
            .matrix_mul_shader_f32()
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
            .collect::<Vec<f64>>();

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
    use crate::fraction_exact::FractionExact;
    use crate::fraction_f64::FractionF64;
    use crate::matrix::ebi_matrix::EbiMatrix;
    use crate::matrix::fraction_matrix_exact::FractionMatrixExact;
    use crate::matrix::fraction_matrix_f64::FractionMatrixF64;
    use crate::matrix::loose_fraction::Type;
    use crate::matrix::mul_gpu::MulGpu;
    use crate::shader::state::COMPUTE_SHADERS;
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
        let m1 = m1.reduce();

        let m2: FractionMatrixExact = vec![
            vec![FractionExact::from(7), FractionExact::from(8)],
            vec![FractionExact::from(9), FractionExact::from(-10)],
            vec![FractionExact::from(-11), FractionExact::from(12)],
        ]
        .try_into()
        .unwrap();
        let m2 = m2.reduce();

        let prod = (&m1).mul_gpu(&m2).unwrap();
        let prod_transformed = (&m1).mul_gpu_transformed(&m2).unwrap();

        assert_eq!(prod.number_of_columns(), 2);
        assert_eq!(prod.number_of_rows(), 2);

        let m3 = vec![
            vec![FractionExact::from(-8), FractionExact::from(24)],
            vec![FractionExact::from(-83), FractionExact::from(154)],
        ];

        assert_eq!(prod.clone().to_vec().unwrap(), m3);
        assert_eq!(prod_transformed.clone().to_vec().unwrap(), m3);
    }

    #[test]
    fn bench_mul_gpu() {
        let repeat = 10;
        let size = 500_usize;

        let mut rng = rand::thread_rng();
        let sqrt = 1000_u64;
        let types = vec![Type::Minus; size * size];
        let numerators = vec![rng.gen_range(0..sqrt); size * size];
        let denominators = vec![rng.gen_range(0..sqrt); size * size];

        let matrices_f64: Vec<FractionMatrixF64> = (0..repeat)
            .into_iter()
            .map(|i| FractionMatrixF64 {
                number_of_columns: size,
                number_of_rows: size,
                values: types
                    .iter()
                    .zip(numerators.iter())
                    .zip(denominators.iter())
                    .enumerate()
                    .map(|(x, ((_typee, nom), den))| {
                        if x == i {
                            FractionF64::from((*nom as i64, den + 1)).0
                        } else {
                            FractionF64::from((*nom as i64, *den)).0
                        }
                    })
                    .collect::<Vec<_>>(),
            })
            .collect();

        let matrices_exact_u64: Vec<FractionMatrixExact> = (0..repeat)
            .into_iter()
            .map(|i| {
                FractionMatrixExact::U64 {
                    number_of_columns: size,
                    number_of_rows: size,
                    types: types.clone(),
                    numerators: numerators.clone(),
                    denominators: denominators
                        .iter()
                        .enumerate()
                        .map(|(x, i)| if x as u64 == *i { i + 1 } else { *i })
                        .collect(),
                }
                .reduce()
            })
            .collect();

        // Init shaders
        {
            let before = Instant::now();
            let _ = COMPUTE_SHADERS.matrix_mul_shader_exact();
            let _ = COMPUTE_SHADERS.matrix_mul_shader_f32();
            let _ = COMPUTE_SHADERS.matrix_mul_shader_u64();
            println!("init shaders:      {:.2?}", before.elapsed());
        }

        // exact u64 gpu
        {
            let before = Instant::now();
            for m in &matrices_exact_u64 {
                let m3 = m.mul_gpu(m).unwrap();

                if !m3.is_exact() {
                    panic!()
                }
            }

            println!("exact u64 gpu:     {:.2?}", before.elapsed());
        }

        // exact u64 gpu transformed
        {
            let before = Instant::now();
            for m in &matrices_exact_u64 {
                let m3 = m.mul_gpu_transformed(m).unwrap();

                if !m3.is_exact() {
                    panic!()
                }
            }

            println!("exact u64 gpu transformed:     {:.2?}", before.elapsed());
        }

        // exact u64
        /*{
            let before = Instant::now();
            for m in &matrices_exact_u64 {
                let m3 = (m * m).unwrap();

                if !m3.is_exact() {
                    panic!()
                }
            }

            println!("exact u64 cpu:     {:.2?}", before.elapsed());
        }*/

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
