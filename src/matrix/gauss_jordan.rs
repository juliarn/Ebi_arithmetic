use anyhow::{Result, anyhow};
use malachite::rational::Rational;
use std::sync::atomic::AtomicBool;

use crate::{
    ebi_matrix::EbiMatrix,
    ebi_number::{One, Zero},
    matrix::{
        fraction_matrix_enum::FractionMatrixEnum, fraction_matrix_exact::FractionMatrixExact,
        fraction_matrix_f64::FractionMatrixF64,
    }, GaussJordan,
};

macro_rules! gauss_jordan {
    ($self:ident) => {
        let number_of_rows = $self.number_of_rows();
        let number_of_columns = $self.number_of_columns();

        if number_of_rows == 0 || number_of_columns == 0 {
            return;
        }

        for row_a in 0..number_of_rows - 1 {
            if $self.values[row_a * number_of_columns + row_a].is_zero() {
                continue;
            } else {
                for row_b in row_a..number_of_rows - 1 {
                    //optimisation: do not attempt to add a factor of 0
                    if !$self.values[(row_b + 1) * number_of_columns + row_a].is_zero() {
                        let mut factor =
                            $self.values[(row_b + 1) * number_of_columns + row_a].clone();
                        factor /= &$self.values[row_a * number_of_columns + row_a];
                        // let factor = &values[row_b + 1][row_a] / &values[row_a][row_a];

                        // println!(
                        //     "\t\t\t\t\tfactor row_a {}, row_b {}, {}",
                        //     row_a, row_b, factor
                        // );
                        for column in row_a..number_of_columns {
                            let mut old = $self.values[row_a * number_of_columns + column].clone();
                            old *= &factor;
                            $self.values[(row_b + 1) * number_of_columns + column] -= old;
                        }

                        // log::debug!("\t\t\t       now {}", self);
                    }
                }
            }
        }

        // println!("row-reduced echelon\n{:?}", values);

        // log::info!("number of columns {}", self.get_number_of_columns());

        // log::info!("first step done");

        for i in (0..number_of_rows).rev() {
            if $self.values[i * number_of_columns + i].is_zero() {
                continue;
            } else {
                for j in (0..i).rev() {
                    let mut factor = $self.values[j * number_of_columns + i].clone();
                    factor /= &$self.values[i * number_of_columns + i];
                    // let factor = &values[j][i] / &values[i][i];

                    for k in i..number_of_columns {
                        let mut old = $self.values[i * number_of_columns + k].clone();
                        old *= &factor;
                        $self.values[j * number_of_columns + k] -= old;
                    }
                }
            }
        }

        // log::debug!("\t\tsecond step        {}", self);

        // log::info!("second step done");
    };
}

macro_rules! gauss_jordan_reduced {
    ($self:expr, $t:ident) => {{
        {
            $self.gauss_jordan();

            let number_of_rows = $self.number_of_rows();
            let number_of_columns = $self.number_of_columns();

            let failed = AtomicBool::new(false);

            $self
                .values
                .chunks_mut(number_of_columns)
                .enumerate()
                .for_each(|(i, row)| {
                    let factor = row[i].clone();
                    if factor.is_zero() {
                        failed.store(true, std::sync::atomic::Ordering::Relaxed);
                    } else {
                        for j in number_of_rows..number_of_columns {
                            row[j] /= &factor;
                        }
                        row[i] = $t::one();
                    }
                });

            if failed.load(std::sync::atomic::Ordering::Relaxed) {
                return Err(anyhow!("matrix has no reduced row-echelon form"));
            }

            // log::info!("third step done");

            Ok($self)
        }
    }};
}

impl GaussJordan for FractionMatrixF64 {
    fn gauss_jordan(&mut self) {
        gauss_jordan!(self);
    }

    fn gauss_jordan_reduced(mut self) -> Result<Self> {
        gauss_jordan_reduced!(self, f64)
    }
}
impl GaussJordan for FractionMatrixExact {
    fn gauss_jordan(&mut self) {
        gauss_jordan!(self);
    }

    fn gauss_jordan_reduced(mut self) -> Result<Self> {
        gauss_jordan_reduced!(self, Rational)
    }
}

impl GaussJordan for FractionMatrixEnum {
    fn gauss_jordan(&mut self) {
        match self {
            FractionMatrixEnum::Approx(m) => m.gauss_jordan(),
            FractionMatrixEnum::Exact(m) => m.gauss_jordan(),
            FractionMatrixEnum::CannotCombineExactAndApprox => {}
        }
    }

    fn gauss_jordan_reduced(self) -> Result<Self> {
        match self {
            FractionMatrixEnum::Approx(m) => {
                Ok(FractionMatrixEnum::Approx(m.gauss_jordan_reduced()?))
            }
            FractionMatrixEnum::Exact(m) => {
                Ok(FractionMatrixEnum::Exact(m.gauss_jordan_reduced()?))
            }
            FractionMatrixEnum::CannotCombineExactAndApprox => {
                return Err(anyhow!("cannot combine exact and approximate arithmetic"));
            }
        }
    }
}
