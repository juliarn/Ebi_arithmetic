use crate::{
    One, Signed,
    ebi_matrix::EbiMatrix,
    ebi_number::Zero,
    fraction::{fraction::EPSILON, fraction_f64::FractionF64},
    pop_front_columns, push_columns,
};
use anyhow::{Error, Result, anyhow};

#[derive(Clone, Debug)]
pub struct FractionMatrixF64 {
    pub(crate) values: Vec<f64>,
    pub(crate) number_of_rows: usize,
    pub(crate) number_of_columns: usize,
}

impl FractionMatrixF64 {
    pub(crate) fn index(&self, row: usize, column: usize) -> usize {
        row * self.number_of_columns + column
    }
}

impl EbiMatrix<FractionF64> for FractionMatrixF64 {
    fn new(number_of_rows: usize, number_of_columns: usize) -> Self {
        Self {
            number_of_rows,
            number_of_columns,
            values: vec![0f64; number_of_rows * number_of_columns],
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
            f64::zero(),
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
            0f64,
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

    fn get(&self, row: usize, column: usize) -> Option<FractionF64> {
        let idx = self.index(row, column);
        Some(FractionF64(*self.values.get(idx)?))
    }

    fn set(&mut self, row: usize, column: usize, value: FractionF64) {
        let idx = self.index(row, column);
        self.values[idx] = value.0;
    }

    fn set_zero(&mut self, row: usize, column: usize) {
        let idx = self.index(row, column);
        self.values[idx] = 0f64;
    }

    fn set_one(&mut self, row: usize, column: usize) {
        let idx = self.index(row, column);
        self.values[idx] = 1f64;
    }

    fn to_vec(self) -> Vec<Vec<FractionF64>> {
        if self.number_of_columns > 0 {
            self.values
                .chunks(self.number_of_columns)
                .map(|x| x.into_iter().map(|f| FractionF64(*f)).collect())
                .collect()
        } else {
            vec![vec![]; self.number_of_rows]
        }
    }

    fn is_one(&self, row: usize, column: usize) -> bool {
        self.values[row * self.number_of_columns + column].is_one()
    }

    fn increase(&mut self, row: usize, column: usize, value: &FractionF64) {
        self.values[row * self.number_of_columns + column] += value.0;
    }

    fn decrease(&mut self, row: usize, column: usize, value: &FractionF64) {
        self.values[row * self.number_of_columns + column] -= value.0;
    }

    fn set_row_zero(&mut self, row: usize) {
        for column in 0..self.number_of_columns {
            self.values[row * self.number_of_columns + column] = 0f64;
        }
    }

    fn is_positive(&self, row: usize, column: usize) -> bool {
        Signed::is_positive(&self.values[row * self.number_of_columns + column])
    }

    fn is_negative(&self, row: usize, column: usize) -> bool {
        Signed::is_negative(&self.values[row * self.number_of_columns + column])
    }
}

impl PartialEq for FractionMatrixF64 {
    fn eq(&self, other: &Self) -> bool {
        self.number_of_columns == other.number_of_columns
            && self.number_of_rows == other.number_of_rows
            && self
                .values
                .iter()
                .zip(other.values.iter())
                .all(|(a, b)| (a - b).abs() < EPSILON)
    }
}

impl Eq for FractionMatrixF64 {}

impl TryFrom<(usize, Vec<FractionF64>)> for FractionMatrixF64 {
    type Error = Error;

    fn try_from(value: (usize, Vec<FractionF64>)) -> Result<Self> {
        let (number_of_columns, values) = value;
        let number_of_rows = values.len() / number_of_columns;

        if number_of_rows * number_of_columns != values.len() {
            return Err(anyhow!("some cells of the matrix are not provided"));
        }

        if number_of_rows != 0 {
            //has rows

            let values = values.into_iter().map(|cell| cell.0).collect::<Vec<_>>();

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

impl TryFrom<Vec<Vec<FractionF64>>> for FractionMatrixF64 {
    type Error = Error;
    fn try_from(values: Vec<Vec<FractionF64>>) -> Result<Self> {
        let number_of_columns = if let Some(x) = values.iter().next() {
            x.len()
        } else {
            0
        };
        let number_of_rows = values.len();

        let mut new_values = Vec::with_capacity(values.len() * number_of_columns);
        for row in values {
            if row.len() != number_of_columns {
                return Err(anyhow!("number of columns is not consistent"));
            }

            for v in row {
                new_values.push(v.0);
            }
        }

        Ok(Self {
            values: new_values,
            number_of_rows,
            number_of_columns,
        })
    }
}

impl std::fmt::Display for FractionMatrixF64 {
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
