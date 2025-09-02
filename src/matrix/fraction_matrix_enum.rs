use std::{
    fmt::{Debug, Display},
    mem,
};

use anyhow::{Error, Result, anyhow};

use crate::{
    ebi_matrix::EbiMatrix,
    exact::MaybeExact,
    exact::{self, is_exact_globally},
    fraction::{
        fraction_enum::FractionEnum, fraction_exact::FractionExact, fraction_f64::FractionF64,
    },
    matrix::{fraction_matrix_exact::FractionMatrixExact, fraction_matrix_f64::FractionMatrixF64},
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FractionMatrixEnum {
    Approx(FractionMatrixF64),
    Exact(FractionMatrixExact),
    CannotCombineExactAndApprox,
}

impl EbiMatrix<FractionEnum> for FractionMatrixEnum {
    fn new(number_of_rows: usize, number_of_columns: usize) -> Self {
        if exact::is_exact_globally() {
            Self::Exact(FractionMatrixExact::new(number_of_rows, number_of_columns))
        } else {
            Self::Approx(FractionMatrixF64::new(number_of_rows, number_of_columns))
        }
    }

    fn number_of_rows(&self) -> usize {
        match self {
            FractionMatrixEnum::Approx(m) => m.number_of_rows(),
            FractionMatrixEnum::Exact(m) => m.number_of_rows(),
            FractionMatrixEnum::CannotCombineExactAndApprox => 0,
        }
    }

    fn number_of_columns(&self) -> usize {
        match self {
            FractionMatrixEnum::Approx(m) => m.number_of_columns(),
            FractionMatrixEnum::Exact(m) => m.number_of_columns(),
            FractionMatrixEnum::CannotCombineExactAndApprox => 0,
        }
    }

    fn push_columns(&mut self, number_of_columns_to_add: usize) {
        match self {
            FractionMatrixEnum::Approx(m) => m.push_columns(number_of_columns_to_add),
            FractionMatrixEnum::Exact(m) => m.push_columns(number_of_columns_to_add),
            FractionMatrixEnum::CannotCombineExactAndApprox => {}
        }
    }

    fn push_rows(&mut self, number_of_rows_to_add: usize) {
        match self {
            FractionMatrixEnum::Approx(m) => m.push_rows(number_of_rows_to_add),
            FractionMatrixEnum::Exact(m) => m.push_rows(number_of_rows_to_add),
            FractionMatrixEnum::CannotCombineExactAndApprox => {}
        }
    }

    fn pop_front_columns(&mut self, number_of_columns_to_remove: usize) {
        match self {
            FractionMatrixEnum::Approx(m) => m.pop_front_columns(number_of_columns_to_remove),
            FractionMatrixEnum::Exact(m) => m.pop_front_columns(number_of_columns_to_remove),
            FractionMatrixEnum::CannotCombineExactAndApprox => {}
        }
    }

    fn get(&self, row: usize, column: usize) -> Option<FractionEnum> {
        Some(match self {
            FractionMatrixEnum::Approx(m) => FractionEnum::Approx(m.get(row, column)?.0),
            FractionMatrixEnum::Exact(m) => FractionEnum::Exact(m.get(row, column)?.0),
            FractionMatrixEnum::CannotCombineExactAndApprox => {
                FractionEnum::CannotCombineExactAndApprox
            }
        })
    }

    fn set(&mut self, row: usize, column: usize, value: FractionEnum) {
        if let FractionMatrixEnum::Approx(m) = self {
            if let FractionEnum::Approx(f) = value {
                m.set(row, column, FractionF64(f));
                return;
            }
        } else if let FractionMatrixEnum::Exact(m) = self {
            if let FractionEnum::Exact(f) = value {
                m.set(row, column, FractionExact(f));
                return;
            }
        }
        mem::swap(self, &mut FractionMatrixEnum::CannotCombineExactAndApprox);
    }

    fn set_zero(&mut self, row: usize, column: usize) {
        match self {
            FractionMatrixEnum::Approx(m) => m.set_zero(row, column),
            FractionMatrixEnum::Exact(m) => m.set_zero(row, column),
            FractionMatrixEnum::CannotCombineExactAndApprox => {}
        }
    }

    fn set_one(&mut self, row: usize, column: usize) {
        match self {
            FractionMatrixEnum::Approx(m) => m.set_one(row, column),
            FractionMatrixEnum::Exact(m) => m.set_one(row, column),
            FractionMatrixEnum::CannotCombineExactAndApprox => {}
        }
    }

    fn is_one(&self, row: usize, column: usize) -> bool {
        match self {
            FractionMatrixEnum::Approx(m) => m.is_one(row, column),
            FractionMatrixEnum::Exact(m) => m.is_one(row, column),
            FractionMatrixEnum::CannotCombineExactAndApprox => false,
        }
    }

    fn to_vec(self) -> Vec<Vec<FractionEnum>> {
        match self {
            FractionMatrixEnum::Approx(m) => m
                .to_vec()
                .into_iter()
                .map(|r| r.into_iter().map(|f| FractionEnum::Approx(f.0)).collect())
                .collect(),
            FractionMatrixEnum::Exact(m) => m
                .to_vec()
                .into_iter()
                .map(|r| r.into_iter().map(|f| FractionEnum::Exact(f.0)).collect())
                .collect(),
            FractionMatrixEnum::CannotCombineExactAndApprox => vec![],
        }
    }

    fn increase(&mut self, row: usize, column: usize, value: &FractionEnum) {
        match self {
            FractionMatrixEnum::Approx(m) => {
                if let Ok(f) = value.approx_ref() {
                    m.values[row * m.number_of_columns + column] += f;
                } else {
                    *self = FractionMatrixEnum::CannotCombineExactAndApprox;
                }
            }
            FractionMatrixEnum::Exact(m) => {
                if let Ok(f) = value.exact_ref() {
                    m.values[row * m.number_of_columns + column] += f;
                } else {
                    *self = FractionMatrixEnum::CannotCombineExactAndApprox;
                }
            }
            FractionMatrixEnum::CannotCombineExactAndApprox => {}
        }
    }

    fn decrease(&mut self, row: usize, column: usize, value: &FractionEnum) {
        match self {
            FractionMatrixEnum::Approx(m) => {
                if let Ok(f) = value.approx_ref() {
                    m.values[row * m.number_of_columns + column] -= f;
                } else {
                    *self = FractionMatrixEnum::CannotCombineExactAndApprox;
                }
            }
            FractionMatrixEnum::Exact(m) => {
                if let Ok(f) = value.exact_ref() {
                    m.values[row * m.number_of_columns + column] -= f;
                } else {
                    *self = FractionMatrixEnum::CannotCombineExactAndApprox;
                }
            }
            FractionMatrixEnum::CannotCombineExactAndApprox => {}
        }
    }

    fn set_row_zero(&mut self, row: usize) {
        match self {
            FractionMatrixEnum::Approx(m) => m.set_row_zero(row),
            FractionMatrixEnum::Exact(m) => m.set_row_zero(row),
            FractionMatrixEnum::CannotCombineExactAndApprox => {}
        }
    }

    fn is_positive(&self, row: usize, column: usize) -> bool {
        match self {
            FractionMatrixEnum::Approx(m) => m.is_positive(row, column),
            FractionMatrixEnum::Exact(m) => m.is_positive(row, column),
            FractionMatrixEnum::CannotCombineExactAndApprox => false,
        }
    }

    fn is_negative(&self, row: usize, column: usize) -> bool {
        match self {
            FractionMatrixEnum::Approx(m) => m.is_negative(row, column),
            FractionMatrixEnum::Exact(m) => m.is_negative(row, column),
            FractionMatrixEnum::CannotCombineExactAndApprox => false,
        }
    }
}

impl TryFrom<Vec<Vec<FractionEnum>>> for FractionMatrixEnum {
    type Error = Error;

    fn try_from(value: Vec<Vec<FractionEnum>>) -> Result<Self> {
        if let Some(x) = value.iter().next() {
            if let Some(y) = x.iter().next() {
                let number_of_columns = x.len();
                //proper matrix
                if y.is_exact() {
                    //exact mode
                    let mut new_rows = Vec::with_capacity(value.len());
                    for row in value {
                        if row.len() != number_of_columns {
                            return Err(anyhow!("number of columns is not consistent"));
                        }

                        let mut new_row = Vec::with_capacity(number_of_columns);
                        for f in row {
                            match f {
                                FractionEnum::Exact(f) => new_row.push(FractionExact(f)),
                                FractionEnum::Approx(_) => {
                                    return Err(anyhow!(
                                        "cannot combine approximate and exact arithmetic"
                                    ));
                                }
                                FractionEnum::CannotCombineExactAndApprox => {
                                    return Err(anyhow!(
                                        "cannot combine approximate and exact arithmetic"
                                    ));
                                }
                            }
                        }
                        new_rows.push(new_row);
                    }

                    let m: FractionMatrixExact = new_rows.try_into()?;
                    Ok(Self::Exact(m))
                } else {
                    //approximate mode
                    let mut new_rows = Vec::with_capacity(value.len());
                    for row in value {
                        if row.len() != number_of_columns {
                            return Err(anyhow!("number of columns is not consistent"));
                        }

                        let mut new_row = Vec::with_capacity(number_of_columns);
                        for f in row {
                            match f {
                                FractionEnum::Exact(_) => {
                                    return Err(anyhow!(
                                        "cannot combine approximate and exact arithmetic"
                                    ));
                                }
                                FractionEnum::Approx(f) => new_row.push(FractionF64(f)),
                                FractionEnum::CannotCombineExactAndApprox => {
                                    return Err(anyhow!(
                                        "cannot combine approximate and exact arithmetic"
                                    ));
                                }
                            }
                        }
                        new_rows.push(new_row);
                    }

                    let m: FractionMatrixF64 = new_rows.try_into()?;
                    Ok(Self::Approx(m))
                }
            } else {
                //rows, no columns
                if is_exact_globally() {
                    let new_rows = vec![vec![]; value.len()];
                    let m: FractionMatrixExact = new_rows.try_into()?;
                    Ok(Self::Exact(m))
                } else {
                    let new_rows = vec![vec![]; value.len()];
                    let m: FractionMatrixF64 = new_rows.try_into()?;
                    Ok(Self::Approx(m))
                }
            }
        } else {
            //no rows
            if is_exact_globally() {
                Ok(Self::Exact(FractionMatrixExact::new(0, 0)))
            } else {
                Ok(Self::Approx(FractionMatrixF64::new(0, 0)))
            }
        }
    }
}

impl Display for FractionMatrixEnum {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Approx(m) => Display::fmt(m, f),
            Self::Exact(m) => Display::fmt(m, f),
            Self::CannotCombineExactAndApprox => {
                write!(f, "cannot combine approximate and exact arithmetic")
            }
        }
    }
}
