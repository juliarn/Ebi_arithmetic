use crate::{
    ebi_number::{One, Zero},
    matrix::{
        ebi_matrix::EbiMatrix,
        fraction_matrix_enum::FractionMatrixEnum,
        fraction_matrix_exact::FractionMatrixExact,
        fraction_matrix_f64::FractionMatrixF64,
        loose_fraction::{self, LooseFraction, Type},
    },
};
use anyhow::{anyhow, Result};
use fraction::Integer;
use itertools::iproduct;
use num::BigUint;
use num_bigint::ToBigUint;
use num_traits::ToPrimitive;
use std::ops::Mul;

impl Mul for &FractionMatrixExact {
    type Output = Result<FractionMatrixExact>;

    fn mul(self, rhs: Self) -> Self::Output {
        if self.number_of_columns() != rhs.number_of_rows() {
            return Err(anyhow!(
                "cannot multiply matrix of size {}x{} with a matrix of size {}x{}",
                self.number_of_rows(),
                self.number_of_columns(),
                rhs.number_of_rows(),
                rhs.number_of_columns()
            ));
        }

        match (self, rhs) {
            (
                FractionMatrixExact::U64 {
                    number_of_columns,
                    types,
                    numerators,
                    denominators,
                    ..
                },
                FractionMatrixExact::U64 {
                    number_of_columns: number_of_columns2,
                    types: types2,
                    numerators: numerators2,
                    denominators: denominators2,
                    ..
                },
            ) => {
                let n = self.number_of_rows();
                let m = self.number_of_columns();
                let p = rhs.number_of_columns();

                let mut new_types = vec![Type::Plus; p * n];
                let mut new_num = vec![0; p * n];
                let mut new_den = vec![1; p * n];

                let mut last_attempted = None;
                'outer: for i in 0..n {
                    for j in 0..p {
                        let idx_ij = i * p + j;
                        for k in 0..m {
                            let idx_ik = i * number_of_columns + k;
                            let idx_kj = k * number_of_columns2 + j;
                            if loose_fraction::checked_add_assign_mul(
                                &mut new_types[idx_ij],
                                &mut new_num[idx_ij],
                                &mut new_den[idx_ij],
                                types[idx_ik],
                                &numerators[idx_ik],
                                &denominators[idx_ik],
                                types2[idx_kj],
                                &numerators2[idx_kj],
                                &denominators2[idx_kj],
                            ) {
                                //no overlflow; continue
                            } else {
                                //overflow detected
                                println!(
                                    "overflow with add_assign_mul {}/{}",
                                    numerators[idx_ik], denominators[idx_ik]
                                );
                                new_types[idx_ij] = Type::Plus;
                                last_attempted = Some((i, j));
                                break 'outer;
                            }
                        }
                    }
                }

                if let Some((r, c)) = last_attempted {
                    //overflow occurred, salvage results and finish it as a larger data type
                    let mut new_new_num = vec![BigUint::zero(); p * n];
                    let mut new_new_den = vec![BigUint::one(); p * n];

                    //first copy the already obtained results to the larger data type
                    (0..n * p).take(r * n + c).for_each(|i| {
                        new_new_num[i] = new_num[i].to_biguint().unwrap();
                        new_new_den[i] = new_den[i].to_biguint().unwrap();
                    });

                    //second, finish the multiplication on the larger data type
                    iproduct!(0..n, 0..p).skip(r * n + c).for_each(|(i, j)| {
                        let idx_ij = i * p + j;
                        for k in 0..m {
                            let idx_ik = i * number_of_columns + k;
                            let idx_kj = k * number_of_columns2 + j;
                            BigUint::add_assign_mul(
                                &mut new_types[idx_ij],
                                &mut new_new_num[idx_ij],
                                &mut new_new_den[idx_ij],
                                types[idx_ik],
                                &numerators[idx_ik].to_biguint().unwrap(),
                                &denominators[idx_ik].to_biguint().unwrap(),
                                types2[idx_kj],
                                &numerators2[idx_kj],
                                &denominators2[idx_kj],
                            )
                        }
                    });

                    Ok(FractionMatrixExact::BigInt {
                        number_of_columns: n,
                        number_of_rows: p,
                        types: new_types,
                        numerators: new_new_num,
                        denominators: new_new_den,
                    })
                } else {
                    //completed normally
                    Ok(FractionMatrixExact::U64 {
                        number_of_columns: n,
                        number_of_rows: p,
                        types: new_types,
                        numerators: new_num,
                        denominators: new_den,
                    })
                }
            }
            (
                FractionMatrixExact::U64 {
                    number_of_columns,
                    types,
                    numerators,
                    denominators,
                    ..
                },
                FractionMatrixExact::BigInt {
                    number_of_columns: number_of_columns2,
                    types: types2,
                    numerators: numerators2,
                    denominators: denominators2,
                    ..
                },
            ) => {
                let n = self.number_of_rows();
                let m = self.number_of_columns();
                let p = rhs.number_of_columns();
                let mut new_types = vec![Type::Plus; p * n];
                let mut new_num = vec![BigUint::zero(); p * n];
                let mut new_den = vec![BigUint::one(); p * n];

                iproduct!(0..n, 0..p).for_each(|(i, j)| {
                    let idx_ij = i * p + j;
                    for k in 0..m {
                        let idx_ik = i * number_of_columns + k;
                        let idx_kj = k * number_of_columns2 + j;
                        BigUint::add_assign_mul(
                            &mut new_types[idx_ij],
                            &mut new_num[idx_ij],
                            &mut new_den[idx_ij],
                            types[idx_ik],
                            &numerators[idx_ik],
                            &denominators[idx_ik],
                            types2[idx_kj],
                            &numerators2[idx_kj],
                            &denominators2[idx_kj],
                        );
                    }
                });

                Ok(FractionMatrixExact::BigInt {
                    number_of_columns: n,
                    number_of_rows: p,
                    types: new_types,
                    numerators: new_num,
                    denominators: new_den,
                })
            }
            (
                FractionMatrixExact::BigInt {
                    number_of_columns,
                    types,
                    numerators,
                    denominators,
                    ..
                },
                FractionMatrixExact::U64 {
                    number_of_columns: number_of_columns2,
                    types: types2,
                    numerators: numerators2,
                    denominators: denominators2,
                    ..
                },
            ) => {
                let n = self.number_of_rows();
                let m = self.number_of_columns();
                let p = rhs.number_of_columns();
                let mut new_types = vec![Type::Plus; p * n];
                let mut new_num = vec![BigUint::zero(); p * n];
                let mut new_den = vec![BigUint::one(); p * n];

                iproduct!(0..n, 0..p).for_each(|(i, j)| {
                    let idx_ij = i * p + j;
                    for k in 0..m {
                        let idx_ik = i * number_of_columns + k;
                        let idx_kj = k * number_of_columns2 + j;
                        BigUint::add_assign_mul(
                            &mut new_types[idx_ij],
                            &mut new_num[idx_ij],
                            &mut new_den[idx_ij],
                            types[idx_ik],
                            &numerators[idx_ik],
                            &denominators[idx_ik],
                            types2[idx_kj],
                            &numerators2[idx_kj],
                            &denominators2[idx_kj],
                        );
                    }
                });

                Ok(FractionMatrixExact::BigInt {
                    number_of_columns: n,
                    number_of_rows: p,
                    types: new_types,
                    numerators: new_num,
                    denominators: new_den,
                })
            }
            (
                FractionMatrixExact::BigInt {
                    number_of_columns,
                    types,
                    numerators,
                    denominators,
                    ..
                },
                FractionMatrixExact::BigInt {
                    number_of_columns: number_of_columns2,
                    types: types2,
                    numerators: numerators2,
                    denominators: denominators2,
                    ..
                },
            ) => {
                let n = self.number_of_rows();
                let m = self.number_of_columns();
                let p = rhs.number_of_columns();
                let mut new_types = vec![Type::Plus; p * n];
                let mut new_num = vec![BigUint::zero(); p * n];
                let mut new_den = vec![BigUint::one(); p * n];

                iproduct!(0..n, 0..p).for_each(|(i, j)| {
                    let idx_ij = i * p + j;
                    for k in 0..m {
                        let idx_ik = i * number_of_columns + k;
                        let idx_kj = k * number_of_columns2 + j;
                        BigUint::add_assign_mul(
                            &mut new_types[idx_ij],
                            &mut new_num[idx_ij],
                            &mut new_den[idx_ij],
                            types[idx_ik],
                            &numerators[idx_ik],
                            &denominators[idx_ik],
                            types2[idx_kj],
                            &numerators2[idx_kj],
                            &denominators2[idx_kj],
                        );
                    }
                });

                Ok(FractionMatrixExact::BigInt {
                    number_of_columns: n,
                    number_of_rows: p,
                    types: new_types,
                    numerators: new_num,
                    denominators: new_den,
                })
            }
        }
    }
}

impl Mul for &FractionMatrixF64 {
    type Output = Result<FractionMatrixF64>;

    fn mul(self, rhs: Self) -> Self::Output {
        if self.number_of_columns() != rhs.number_of_rows() {
            return Err(anyhow!(
                "cannot multiply matrix of size {}x{} with a matrix of size {}x{}",
                self.number_of_rows(),
                self.number_of_columns(),
                rhs.number_of_rows(),
                rhs.number_of_columns()
            ));
        }

        let n = self.number_of_rows();
        let m = self.number_of_columns();
        let p = rhs.number_of_columns();
        let mut values = vec![0f64; p * n];

        iproduct!(0..n, 0..p).for_each(|(i, j)| {
            for k in 0..m {
                let idx_ik = self.index(i, k);
                let idx_kj = rhs.index(k, j);
                values[i * n + j] += self.values[idx_ik] * rhs.values[idx_kj];
            }
        });

        Ok(FractionMatrixF64 {
            values,
            number_of_columns: n,
            number_of_rows: p,
        })
    }
}

impl Mul for &FractionMatrixEnum {
    type Output = Result<FractionMatrixEnum>;

    fn mul(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (FractionMatrixEnum::Approx(m1), FractionMatrixEnum::Approx(m2)) => {
                Ok(FractionMatrixEnum::Approx((m1 * m2)?))
            }
            (FractionMatrixEnum::Exact(m1), FractionMatrixEnum::Exact(m2)) => {
                Ok(FractionMatrixEnum::Exact((m1 * m2)?))
            }
            _ => Ok(FractionMatrixEnum::CannotCombineExactAndApprox),
        }
    }
}

#[cfg(test)]
mod tests {
    use num_bigint::ToBigUint;
    use rand::Rng;
    use std::time::Instant;

    use crate::matrix::fraction_matrix_exact::FractionMatrixExact;
    use crate::matrix::fraction_matrix_f64::FractionMatrixF64;
    use crate::{
        ebi_number::{One, Zero},
        exact::MaybeExact,
        f, f0, f1,
        fraction::Fraction,
        fraction_f64::FractionF64,
        matrix::{ebi_matrix::EbiMatrix, fraction_matrix::FractionMatrix, loose_fraction::Type},
    };

    #[test]
    fn fraction_matrix_mul() {
        let m1: FractionMatrix = vec![vec![f!(1), f!(2), f!(3)], vec![f!(4), f!(5), f!(6)]]
            .try_into()
            .unwrap();

        (&m1 * &m1).unwrap_err();

        let m2: FractionMatrix = vec![
            vec![f!(7), f!(8)],
            vec![f!(9), f!(10)],
            vec![f!(11), f!(12)],
        ]
        .try_into()
        .unwrap();

        (&m2 * &m2).unwrap_err();

        let prod = (&m1 * &m2).unwrap();

        assert_eq!(prod.number_of_columns(), 2);
        assert_eq!(prod.number_of_rows(), 2);

        let m3 = vec![vec![f!(58), f!(64)], vec![f!(139), f!(154)]];

        assert_eq!(prod.clone().to_vec().unwrap(), m3);

        let m2 = m2.reduce();

        let prod = (&m1 * &m2).unwrap();
        assert_eq!(prod.to_vec().unwrap(), m3);

        let m1 = m1.reduce();

        let prod = (&m1 * &m2).unwrap();
        assert_eq!(prod.to_vec().unwrap(), m3);
    }

    #[test]
    fn fraction_matrix_mul_nan() {
        let m1 = vec![vec![
            Fraction::infinity(),
            Fraction::neg_infinity(),
            f!(-8, 3),
        ]];
        let m1: FractionMatrix = m1.try_into().unwrap();

        (&m1 * &m1).unwrap_err();

        let m2: FractionMatrix = vec![vec![f0!()], vec![f1!()], vec![f!(-8, 3)]]
            .try_into()
            .unwrap();

        (&m2 * &m2).unwrap_err();

        let prod = (&m1 * &m2).unwrap();

        assert_eq!(prod.number_of_columns(), 1);
        assert_eq!(prod.number_of_rows(), 1);

        let m3 = vec![vec![Fraction::nan()]];

        assert_eq!(prod.clone().to_vec().unwrap(), m3);
    }

    #[test]
    fn fraction_matrix_mul_overflow_1() {
        let m1: FractionMatrix = vec![vec![f!(u64::MAX), f!(2), f!(3)], vec![f!(4), f!(5), f!(6)]]
            .try_into()
            .unwrap();
        let m1 = m1.reduce();

        let m2: FractionMatrix = vec![
            vec![f!(u64::MAX), f!(8)],
            vec![f!(9), f!(10)],
            vec![f!(11), f!(12)],
        ]
        .try_into()
        .unwrap();
        let m2 = m2.reduce();

        let prod = (&m1 * &m2).unwrap();

        let m3 = vec![
            [
                "340282366920938463426481119284349108276".parse().unwrap(),
                "147573952589676412976".parse().unwrap(),
            ],
            ["73786976294838206571".parse().unwrap(), f!(154)],
        ];

        assert_eq!(prod.to_vec().unwrap(), m3);
    }

    #[test]
    fn fraction_matrix_mul_overflow_2() {
        let m1: FractionMatrix = vec![vec![f!(u64::MAX), f!(2), f!(3)], vec![f!(4), f!(5), f!(6)]]
            .try_into()
            .unwrap();
        let m1 = m1.reduce();

        let m2: FractionMatrix = vec![
            vec![f!(1), f!(8)],
            vec![f!(9), f!(10)],
            vec![f!(11), f!(12)],
        ]
        .try_into()
        .unwrap();
        let m2 = m2.reduce();

        let prod = (&m1 * &m2).unwrap();

        let m3 = vec![
            [
                "18446744073709551666".parse().unwrap(),
                "147573952589676412976".parse().unwrap(),
            ],
            [f!(115), f!(154)],
        ];

        assert_eq!(prod.to_vec().unwrap(), m3);
    }

    #[test]
    fn fraction_matrix_mul_overflow_3() {
        let m1: FractionMatrix = vec![vec![-f!(u64::MAX), f!(2), f!(3)], vec![f!(4), f!(5), f!(6)]]
            .try_into()
            .unwrap();
        let m1 = m1.reduce();

        let m2: FractionMatrix = vec![
            vec![f!(1), f!(8)],
            vec![f!(9), f!(10)],
            vec![f!(11), f!(12)],
        ]
        .try_into()
        .unwrap();
        let m2 = m2.reduce();

        let prod = (&m1 * &m2).unwrap();

        let m3 = vec![
            [
                "-18446744073709551564".parse().unwrap(),
                "-147573952589676412864".parse().unwrap(),
            ],
            [f!(115), f!(154)],
        ];

        assert_eq!(prod.to_vec().unwrap(), m3);
    }

    #[test]
    fn fraction_matrix_mul_overflow_4() {
        let m1: FractionMatrix = vec![vec![-f!(u64::MAX), f!(2), f!(3)], vec![f!(4), f!(5), f!(6)]]
            .try_into()
            .unwrap();
        // let m1 = m1.reduce();

        let m2: FractionMatrix = vec![
            vec![f!(u64::MAX), f!(8)],
            vec![f!(9), f!(10)],
            vec![f!(11), f!(12)],
        ]
        .try_into()
        .unwrap();
        let m2 = m2.reduce();

        let prod = (&m1 * &m2).unwrap();

        let m3 = vec![
            vec![
                "-340282366920938463426481119284349108174".parse().unwrap(),
                "-147573952589676412864".parse().unwrap(),
            ],
            vec!["73786976294838206571".parse().unwrap(), f!(154)],
        ];

        assert_eq!(prod.to_vec().unwrap(), m3);
    }

    #[test]
    fn bench_mul() {
        let repeat = 5;
        let size = 100_usize;

        let mut rng = rand::thread_rng();
        let sqrt = 100_u64;
        let types = vec![Type::Plus; size * size];
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

        let matrices_exact_bigint: Vec<FractionMatrixExact> = (0..repeat)
            .into_iter()
            .map(|i| FractionMatrixExact::BigInt {
                number_of_columns: size,
                number_of_rows: size,
                types: types.clone(),
                numerators: numerators.iter().map(|i| i.to_biguint().unwrap()).collect(),
                denominators: denominators
                    .iter()
                    .enumerate()
                    .map(|(x, i)| {
                        if x as u64 == *i {
                            (i + 1).to_biguint().unwrap()
                        } else {
                            i.to_biguint().unwrap()
                        }
                    })
                    .collect(),
            })
            .collect();

        let matrices_exact_u64: Vec<FractionMatrixExact> = (0..repeat)
            .into_iter()
            .map(|i| FractionMatrixExact::U64 {
                number_of_columns: size,
                number_of_rows: size,
                types: types.clone(),
                numerators: numerators.clone(),
                denominators: denominators
                    .iter()
                    .enumerate()
                    .map(|(x, i)| if x as u64 == *i { i + 1 } else { *i })
                    .collect(),
            })
            .collect();

        //exact biguint
        {
            let before = Instant::now();
            for m in matrices_exact_bigint {
                let m3 = (&m * &m).unwrap();

                if !m3.is_exact() {
                    panic!()
                }
            }

            println!("exact BigUint: {:.2?}", before.elapsed());
        }

        //exact u64
        {
            let before = Instant::now();
            for m in matrices_exact_u64 {
                let m3 = (&m * &m).unwrap();

                if !m3.is_exact() {
                    panic!()
                }
            }

            println!("exact u64:     {:.2?}", before.elapsed());
        }

        //f64
        {
            let before = Instant::now();
            for m in matrices_f64 {
                let m3 = (&m * &m).unwrap();

                if m3.is_exact() {
                    panic!()
                }
            }

            println!("approx f64:    {:.2?}", before.elapsed());
        }
    }
}
