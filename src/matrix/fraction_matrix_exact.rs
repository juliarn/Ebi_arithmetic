use anyhow::{Error, Result, anyhow};
use itertools::Itertools;
use malachite::{
    base::num::basic::traits::{One as MOne, Zero as MZero},
    rational::Rational,
};

use crate::{
    One, Signed, Zero, ebi_matrix::EbiMatrix, fraction::fraction_exact::FractionExact,
    pop_front_columns, push_columns,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FractionMatrixExact {
    pub(crate) values: Vec<Rational>,
    pub(crate) number_of_rows: usize,
    pub(crate) number_of_columns: usize,
}

impl FractionMatrixExact {
    pub(crate) fn index(&self, row: usize, column: usize) -> usize {
        row * self.number_of_columns + column
    }
}

impl EbiMatrix<FractionExact> for FractionMatrixExact {
    fn new(number_of_rows: usize, number_of_columns: usize) -> Self {
        Self {
            number_of_rows,
            number_of_columns,
            values: vec![Rational::zero(); number_of_rows * number_of_columns],
        }
    }

    fn number_of_rows(&self) -> usize {
        self.number_of_rows
    }

    fn number_of_columns(&self) -> usize {
        self.number_of_columns
    }

    fn push_columns(&mut self, number_of_columns_to_add: usize) {
        push_columns!(
            Rational::ZERO,
            number_of_columns_to_add,
            self.values,
            self.number_of_rows,
            self.number_of_columns
        );
        self.number_of_columns += number_of_columns_to_add;
    }

    fn push_rows(&mut self, number_of_rows_to_add: usize) {
        self.values.resize(
            self.values.len() + number_of_rows_to_add * self.number_of_columns,
            Rational::zero(),
        );
        self.number_of_rows += number_of_rows_to_add;
    }

    fn pop_front_columns(&mut self, number_of_columns_to_remove: usize) {
        pop_front_columns!(
            number_of_columns_to_remove,
            self.values,
            self.number_of_rows,
            self.number_of_columns
        );
        self.number_of_columns -= number_of_columns_to_remove;
    }

    fn get(&self, row: usize, column: usize) -> Option<FractionExact> {
        let idx = self.index(row, column);
        Some(FractionExact(self.values.get(idx)?.clone()))
    }

    fn set(&mut self, row: usize, column: usize, value: FractionExact) {
        self.values[row * self.number_of_columns + column] = value.0;
    }

    fn set_zero(&mut self, row: usize, column: usize) {
        self.values[row * self.number_of_columns + column] = Rational::ZERO;
    }

    fn is_one(&self, row: usize, column: usize) -> bool {
        self.values[row * self.number_of_columns + column].is_one()
    }

    fn set_one(&mut self, row: usize, column: usize) {
        self.values[row * self.number_of_columns + column] = Rational::ONE;
    }

    fn to_vec(self) -> Vec<Vec<FractionExact>> {
        if self.number_of_columns > 0 {
            self.values
                .into_iter()
                .chunks(self.number_of_columns)
                .into_iter()
                .map(|x| x.into_iter().map(|f| FractionExact(f)).collect())
                .collect()
        } else {
            vec![vec![]; self.number_of_rows]
        }
    }

    fn increase(&mut self, row: usize, column: usize, value: &FractionExact) {
        self.values[row * self.number_of_columns + column] += &value.0
    }

    fn decrease(&mut self, row: usize, column: usize, value: &FractionExact) {
        self.values[row * self.number_of_columns + column] -= &value.0
    }

    fn set_row_zero(&mut self, row: usize) {
        for column in 0..self.number_of_columns {
            self.values[row * self.number_of_columns + column] = Rational::zero();
        }
    }

    fn is_positive(&self, row: usize, column: usize) -> bool {
        Signed::is_positive(&self.values[row * self.number_of_columns + column])
    }

    fn is_negative(&self, row: usize, column: usize) -> bool {
        Signed::is_negative(&self.values[row * self.number_of_columns + column])
    }
}

impl TryFrom<Vec<Vec<FractionExact>>> for FractionMatrixExact {
    type Error = Error;

    fn try_from(value: Vec<Vec<FractionExact>>) -> Result<Self> {
        let number_of_rows = value.len();
        if let Some(x) = value.iter().next() {
            let number_of_columns = x.len();
            //has rows

            let mut values = Vec::with_capacity(number_of_rows * number_of_columns);
            for row in value.into_iter() {
                if row.len() != number_of_columns {
                    return Err(anyhow!("number of columns is not consistent"));
                }

                values.extend(row.into_iter().map(|f| f.0));
            }

            Ok(Self {
                number_of_columns,
                number_of_rows,
                values,
            })
        } else {
            //no rows
            Ok(Self {
                number_of_columns: 0,
                number_of_rows: 0,
                values: vec![],
            })
        }
    }
}

impl std::fmt::Display for FractionMatrixExact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{{{{")?;
        if self.number_of_columns > 0 {
            for (i, row) in self.values.chunks(self.number_of_columns).enumerate() {
                for (j, fraction) in row.iter().enumerate() {
                    write!(f, "{}", fraction.to_string())?;
                    if j < row.len() - 1 {
                        write!(f, ", ")?;
                    }
                }
                if i < self.number_of_rows - 1 {
                    write!(f, "}},\n {{")?;
                }
            }
        } else {
            for _ in 0..self.number_of_rows {
                write!(f, "}},\n{{")?;
            }
        }
        write!(f, "}}}}")
    }
}
