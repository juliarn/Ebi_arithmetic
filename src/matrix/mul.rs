use anyhow::{Result, anyhow};
use itertools::iproduct;
use malachite::rational::Rational;
use std::ops::Mul;

use crate::{
    EbiMatrix, MaybeExact, Zero,
    fraction::{
        fraction_enum::FractionEnum, fraction_exact::FractionExact, fraction_f64::FractionF64,
    },
    matrix::{
        fraction_matrix_enum::FractionMatrixEnum, fraction_matrix_exact::FractionMatrixExact,
        fraction_matrix_f64::FractionMatrixF64,
    },
};

macro_rules! mul_mat_mat {
    ($t:ident, $u:ident, $v:ident) => {
        impl Mul for &$t {
            type Output = Result<$t>;

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

                let result_rows = self.number_of_rows();
                let result_columns = rhs.number_of_columns();
                let mut result = vec![$v::zero(); result_rows * result_columns];

                iproduct!(0..result_rows, 0..result_columns).for_each(|(row, column)| {
                    for k in 0..self.number_of_columns() {
                        result[row * result_columns + column] +=
                            &self.values[row * self.number_of_columns() + k] * &rhs.values[k * rhs.number_of_columns() + column];
                    }
                });

                Ok($t {
                    values: result,
                    number_of_columns: result_columns,
                    number_of_rows: result_rows,
                })
            }
        }
    };
}

macro_rules! mul_vec_mat {
    ($t:ident, $u:ident, $v:ident) => {
        impl Mul<&$t> for &Vec<$u> {
            type Output = Result<Vec<$u>>;

            fn mul(self, rhs: &$t) -> Self::Output {
                if self.len() != rhs.number_of_rows() {
                    return Err(anyhow!(
                        "cannot multiply a vector of size {} with a matrix of size {}x{}",
                        self.len(),
                        rhs.number_of_rows(),
                        rhs.number_of_columns(),
                    ));
                }

                let mut result = vec![$v::zero(); rhs.number_of_columns()];
                for row in 0..rhs.number_of_rows() {
                    for column in 0..rhs.number_of_columns() {
                        result[column] +=
                            &rhs.values[row * rhs.number_of_columns() + column] * &self[row].0;
                    }
                }
                Ok(result.into_iter().map(|f| $u(f)).collect())
            }
        }
    };
}

macro_rules! mul_mat_vec {
    ($t:ident, $u:ident, $v:ident) => {
        impl Mul<&Vec<$u>> for &$t {
            type Output = Result<Vec<$u>>;

            fn mul(self, rhs: &Vec<$u>) -> Self::Output {
                if self.number_of_columns() != rhs.len() {
                    return Err(anyhow!(
                        "cannot multiply matrix of size {}x{} with a vector of size {}",
                        self.number_of_rows(),
                        self.number_of_columns(),
                        rhs.len(),
                    ));
                }

                let mut result = vec![$v::zero(); self.number_of_rows()];
                for row in 0..self.number_of_rows() {
                    for column in 0..self.number_of_columns() {
                        result[row] +=
                            &self.values[row * self.number_of_columns() + column] * &rhs[column].0;
                    }
                }
                Ok(result.into_iter().map(|f| $u(f)).collect())
            }
        }
    };
}

// ===================== f64 =====================

mul_mat_mat!(FractionMatrixF64, FractionF64, f64);
mul_vec_mat!(FractionMatrixF64, FractionF64, f64);
mul_mat_vec!(FractionMatrixF64, FractionF64, f64);

// ===================== exact =====================

mul_mat_mat!(FractionMatrixExact, FractionExact, Rational);
mul_vec_mat!(FractionMatrixExact, FractionExact, Rational);
mul_mat_vec!(FractionMatrixExact, FractionExact, Rational);

// ===================== enum =====================

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

impl Mul<&Vec<FractionEnum>> for &FractionMatrixEnum {
    type Output = Result<Vec<FractionEnum>>;

    fn mul(self, rhs: &Vec<FractionEnum>) -> Self::Output {
        if self.number_of_columns() != rhs.len() {
            return Err(anyhow!(
                "cannot multiply matrix of size {}x{} with a vector of size {}",
                self.number_of_rows(),
                self.number_of_columns(),
                rhs.len(),
            ));
        }

        match self {
            FractionMatrixEnum::Approx(m) => {
                let mut result = vec![f64::zero(); self.number_of_rows()];
                for row in 0..m.number_of_rows() {
                    for column in 0..m.number_of_columns() {
                        result[row] += &m.values[row * m.number_of_columns() + column]
                            * rhs[column].approx_ref()?;
                    }
                }
                Ok(result
                    .into_iter()
                    .map(|f| FractionEnum::Approx(f))
                    .collect())
            }
            FractionMatrixEnum::Exact(m) => {
                let mut result = vec![Rational::zero(); self.number_of_rows()];
                for row in 0..m.number_of_rows() {
                    for column in 0..m.number_of_columns() {
                        result[row] += &m.values[row * m.number_of_columns() + column]
                            * rhs[column].exact_ref()?;
                    }
                }
                Ok(result.into_iter().map(|f| FractionEnum::Exact(f)).collect())
            }
            FractionMatrixEnum::CannotCombineExactAndApprox => {
                Err(anyhow!("cannot combine exact and approximate arithmetic"))
            }
        }
    }
}

impl Mul<&FractionMatrixEnum> for &Vec<FractionEnum> {
    type Output = Result<Vec<FractionEnum>>;

    fn mul(self, rhs: &FractionMatrixEnum) -> Self::Output {
        if self.len() != rhs.number_of_rows() {
            return Err(anyhow!(
                "cannot multiply a vector of size {} with a matrix of size {}x{}",
                self.len(),
                rhs.number_of_rows(),
                rhs.number_of_columns(),
            ));
        }

        match rhs {
            FractionMatrixEnum::Approx(m) => {
                let mut result = vec![f64::zero(); m.number_of_columns()];
                for row in 0..m.number_of_rows() {
                    for column in 0..m.number_of_columns() {
                        result[column] += &m.values[row * m.number_of_columns() + column]
                            * self[row].approx_ref()?;
                    }
                }
                Ok(result
                    .into_iter()
                    .map(|f| FractionEnum::Approx(f))
                    .collect())
            }
            FractionMatrixEnum::Exact(m) => {
                let mut result = vec![Rational::zero(); m.number_of_columns()];
                for row in 0..m.number_of_rows() {
                    for column in 0..m.number_of_columns() {
                        result[column] += &m.values[row * m.number_of_columns() + column]
                            * self[row].exact_ref()?;
                    }
                }
                Ok(result.into_iter().map(|f| FractionEnum::Exact(f)).collect())
            }
            FractionMatrixEnum::CannotCombineExactAndApprox => {
                Err(anyhow!("cannot combine approximate and exact arithmetic"))
            }
        }
    }
}

#[cfg(test)]
mod tests {

    use crate::{
        EbiMatrix, MaybeExact,
        fraction::{fraction::Fraction, fraction_enum::FractionEnum},
        matrix::fraction_matrix_enum::FractionMatrixEnum,
        set_exact_globally,
    };
    use std::time::Instant;

    use anyhow::Result;
    use rand::Rng;
    use serial_test::serial;

    use crate::{
        f,
        fraction::{fraction_exact::FractionExact, fraction_f64::FractionF64},
        matrix::{
            fraction_matrix::FractionMatrix, fraction_matrix_exact::FractionMatrixExact,
            fraction_matrix_f64::FractionMatrixF64,
        },
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

        assert_eq!(prod.clone().to_vec(), m3);

        let prod = (&m1 * &m2).unwrap();
        assert_eq!(prod.to_vec(), m3);

        let prod = (&m1 * &m2).unwrap();
        assert_eq!(prod.to_vec(), m3);
    }

    #[test]
    fn fraction_matrix_mul_overflow_1() {
        let m1: FractionMatrix = vec![vec![f!(u64::MAX), f!(2), f!(3)], vec![f!(4), f!(5), f!(6)]]
            .try_into()
            .unwrap();

        let m2: FractionMatrix = vec![
            vec![f!(u64::MAX), f!(8)],
            vec![f!(9), f!(10)],
            vec![f!(11), f!(12)],
        ]
        .try_into()
        .unwrap();

        let prod = (&m1 * &m2).unwrap();

        let m3 = vec![
            [
                "340282366920938463426481119284349108276".parse().unwrap(),
                "147573952589676412976".parse().unwrap(),
            ],
            ["73786976294838206571".parse().unwrap(), f!(154)],
        ];

        assert_eq!(prod.to_vec(), m3);
    }

    #[test]
    fn fraction_matrix_mul_overflow_2() {
        let m1: FractionMatrix = vec![vec![f!(u64::MAX), f!(2), f!(3)], vec![f!(4), f!(5), f!(6)]]
            .try_into()
            .unwrap();

        let m2: FractionMatrix = vec![
            vec![f!(1), f!(8)],
            vec![f!(9), f!(10)],
            vec![f!(11), f!(12)],
        ]
        .try_into()
        .unwrap();

        let prod = (&m1 * &m2).unwrap();

        let m3 = vec![
            [
                "18446744073709551666".parse().unwrap(),
                "147573952589676412976".parse().unwrap(),
            ],
            [f!(115), f!(154)],
        ];

        assert_eq!(prod.to_vec(), m3);
    }

    #[test]
    fn fraction_matrix_mul_overflow_3() {
        let m1: FractionMatrix = vec![vec![-f!(u64::MAX), f!(2), f!(3)], vec![f!(4), f!(5), f!(6)]]
            .try_into()
            .unwrap();

        let m2: FractionMatrix = vec![
            vec![f!(1), f!(8)],
            vec![f!(9), f!(10)],
            vec![f!(11), f!(12)],
        ]
        .try_into()
        .unwrap();

        let prod = (&m1 * &m2).unwrap();

        let m3 = vec![
            [
                "-18446744073709551564".parse().unwrap(),
                "-147573952589676412864".parse().unwrap(),
            ],
            [f!(115), f!(154)],
        ];

        assert_eq!(prod.to_vec(), m3);
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

        let prod = (&m1 * &m2).unwrap();

        let m3 = vec![
            vec![
                "-340282366920938463426481119284349108174".parse().unwrap(),
                "-147573952589676412864".parse().unwrap(),
            ],
            vec!["73786976294838206571".parse().unwrap(), f!(154)],
        ];

        assert_eq!(prod.to_vec(), m3);
    }

    // #[test]
    fn _bench_mul() {
        let repeat = 5;
        let size = 100_usize;

        let mut rng = rand::rng();
        let sqrt = 1000_u64;
        let numerators = vec![rng.random_range(0..sqrt); size * size];
        let denominators = vec![rng.random_range(0..sqrt); size * size];

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

        //exact
        {
            let before = Instant::now();
            for m in matrices_exact {
                let m3 = (&m * &m).unwrap();

                if !m3.is_exact() {
                    panic!()
                }
            }

            println!("exact:         {:.2?}", before.elapsed());
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

    #[test]
    fn matrix_vector_multiplication_enum_exact() {
        let m: FractionMatrixEnum = vec![
            vec![6.into(), 2.into(), 4.into()],
            vec![(-1).into(), 4.into(), 3.into()],
            vec![(-2).into(), 9.into(), 3.into()],
        ]
        .try_into()
        .unwrap();
        let v: Vec<FractionEnum> = vec![4.into(), (-2).into(), 1.into()];

        let x = (&m * &v).unwrap();

        let t = vec![24.into(), (-9).into(), (-23).into()];

        assert_eq!(x, t);
    }

    #[test]
    #[serial]
    fn matrix_vector_multiplication_enum_approx() {
        set_exact_globally(false);
        let m: FractionMatrixEnum = vec![
            vec![6.into(), 2.into(), 4.into()],
            vec![(-1).into(), 4.into(), 3.into()],
            vec![(-2).into(), 9.into(), 3.into()],
        ]
        .try_into()
        .unwrap();
        let v: Vec<FractionEnum> = vec![4.into(), (-2).into(), 1.into()];

        let x = (&m * &v).unwrap();

        let t = vec![24.into(), (-9).into(), (-23).into()];
        set_exact_globally(true);

        assert_eq!(x, t);
    }

    #[test]
    fn matrix_vector_multiplication_approx() {
        let m: FractionMatrixF64 = vec![
            vec![6.into(), 2.into(), 4.into()],
            vec![(-1).into(), 4.into(), 3.into()],
            vec![(-2).into(), 9.into(), 3.into()],
        ]
        .try_into()
        .unwrap();
        let v: Vec<FractionF64> = vec![4.into(), (-2).into(), 1.into()];

        let x = (&m * &v).unwrap();

        let t = vec![24.into(), (-9).into(), (-23).into()];

        assert_eq!(x, t);
    }

    #[test]
    fn matrix_vector_multiplication_exact() {
        let m: FractionMatrixExact = vec![
            vec![6.into(), 2.into(), 4.into()],
            vec![(-1).into(), 4.into(), 3.into()],
            vec![(-2).into(), 9.into(), 3.into()],
        ]
        .try_into()
        .unwrap();
        let v: Vec<FractionExact> = vec![4.into(), (-2).into(), 1.into()];

        let x = (&m * &v).unwrap();

        let t = vec![24.into(), (-9).into(), (-23).into()];

        assert_eq!(x, t);
    }

    #[test]
    fn mul_small() {
        //exact
        let m: FractionMatrixExact = vec![vec![0.into(), 1.into()], vec![0.into(), 1.into()]]
            .try_into()
            .unwrap();

        let v: Vec<FractionExact> = vec![1.into(), 0.into()];

        let answer_mv = vec![0.into(), 0.into()];
        let answer_vm: Vec<FractionExact> = vec![0.into(), 1.into()];

        assert_eq!((&m * &v).unwrap(), answer_mv);
        assert_eq!((&v * &m).unwrap(), answer_vm);

        //f64
        let m: FractionMatrixF64 = vec![vec![0.into(), 1.into()], vec![0.into(), 1.into()]]
            .try_into()
            .unwrap();

        let v: Vec<FractionF64> = vec![1.into(), 0.into()];

        let answer_mv = vec![0.into(), 0.into()];
        let answer_vm = vec![0.into(), 1.into()];

        assert_eq!((&m * &v).unwrap(), answer_mv);
        assert_eq!((&v * &m).unwrap(), answer_vm);

        //enum
        let m: FractionMatrixEnum = vec![vec![0.into(), 1.into()], vec![0.into(), 1.into()]]
            .try_into()
            .unwrap();

        let v: Vec<FractionEnum> = vec![1.into(), 0.into()];

        let answer_mv = vec![0.into(), 0.into()];
        let answer_vm = vec![0.into(), 1.into()];

        assert_eq!((&m * &v).unwrap(), answer_mv);
        assert_eq!((&v * &m).unwrap(), answer_vm);
    }

    #[test]
    fn mul_vector_matrix() {
        let m: FractionMatrixExact = vec![
            vec![0.into(), 1.into(), 2.into()],
            vec![0.into(), 1.into(), 2.into()],
        ]
        .try_into()
        .unwrap();

        let v: Vec<FractionExact> = vec![0.into(), 1.into()];
        let v2: Vec<FractionExact> = vec![0.into(), 1.into(), 2.into()];

        let answer_mv = vec![5.into(), 5.into()];
        let answer_vm: Vec<FractionExact> = vec![0.into(), 1.into(), 2.into()];

        assert_eq!((&m * &v2).unwrap(), answer_mv);
        assert_eq!((&v * &m).unwrap(), answer_vm);
    }

    fn convert(values: Vec<Vec<f64>>) -> Result<FractionMatrixF64> {
        values
            .into_iter()
            .map(|r| r.into_iter().map(|x| FractionF64(x)).collect())
            .collect::<Vec<Vec<FractionF64>>>()
            .try_into()
    }

    #[test]
    fn mul_big() {
        let a = convert(vec![
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.2, 0.0],
            vec![0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.2, 0.2, 0.0, 0.0, 0.0],
            vec![0.0, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ])
        .unwrap();

        let f = convert(vec![
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.6],
            vec![0.0, 0.44999999999999996, 1.0, 0.0, 0.75, 0.27],
            vec![0.6, 0.0, 0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.6, 0.0, 0.0, 1.0, 0.36],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ])
        .unwrap();

        let fa = convert(vec![
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.2, 0.6],
            vec![0.0, 0.15, 0.15, 0.25, 0.0, 0.09, 0.09, 0.27],
            vec![0.0, 0.0, 0.6, 0.2, 0.2, 0.0, 0.0, 0.0],
            vec![0.0, 0.2, 0.2, 0.0, 0.0, 0.12, 0.12, 0.36],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ])
        .unwrap();

        println!("f\n{}", f);
        println!("a\n{}", a);

        println!("f*a\n{}", (&f * &a).unwrap());
        println!("f*a\n{}", fa);

        assert_eq!((&f * &a).unwrap(), fa);
    }
}
