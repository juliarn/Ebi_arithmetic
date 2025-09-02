use std::mem;

use crate::{
    EbiMatrix, GaussJordan, Inversion, One, Recip, Zero,
    matrix::{
        fraction_matrix_enum::FractionMatrixEnum, fraction_matrix_exact::FractionMatrixExact,
        fraction_matrix_f64::FractionMatrixF64,
    },
};
use anyhow::{Result, anyhow};
use malachite::rational::Rational;

macro_rules! invert {
    ($self:expr, $t:ident) => {{
        if $self.number_of_columns() != $self.number_of_rows() {
            return Err(anyhow!("can only take the inverse of a square matrix"));
        }

        //optimisation: size-zero matrix
        if $self.number_of_rows().is_zero() {
            return Ok($self);
        }

        //optimisation: size-one matrix
        if $self.number_of_rows().is_one() {
            if $self.values[0].is_zero() {
                return Err(anyhow!("matrix is not invertible"));
            }

            $self.values[0] = $self.values[0].clone().recip();
            return Ok($self);
        }

        //optimisation: size-two matrix
        if $self.number_of_rows() == 2 {
            //compute determinant
            let mut det = $self.values[0].clone();
            det *= &$self.values[3];
            let mut det2 = $self.values[1].clone();
            det2 *= &$self.values[2];
            det -= det2;

            if det.is_zero() {
                return Err(anyhow!("matrix is not invertible"));
            }

            // log::debug!("determinant {}", det);

            det = det.recip();

            //perform inverse
            let (m1, m2) = $self.values.split_at_mut(2);
            mem::swap(&mut m1[0], &mut m2[1]);

            $self.values[0] *= &det;
            $self.values[3] *= &det;

            $self.values[2] *= -&det;
            $self.values[1] *= -det;
            return Ok($self);
        }

        // println!("compute inverse of\n{}", $self);

        //extend the rows with the identity matrix
        $self.push_columns($self.number_of_rows);
        for i in 0..$self.number_of_rows {
            let idx_ii = $self.index(i, $self.number_of_rows + i);
            $self.values[idx_ii] = $t::one();
        }

        // println!("add identity\n{}", self);

        //solve
        $self = $self.gauss_jordan_reduced()?;

        // println!("solved\n{}", self);

        //remove the columns
        $self.pop_front_columns($self.number_of_rows);

        // log::info!("inverse done");

        Ok($self)
    }};
}

impl Inversion for FractionMatrixF64 {
    fn invert(mut self) -> Result<Self> {
        invert!(self, f64)
    }
}

impl Inversion for FractionMatrixExact {
    fn invert(mut self) -> Result<Self> {
        invert!(self, Rational)
    }
}

impl Inversion for FractionMatrixEnum {
    fn invert(self) -> Result<Self>
    where
        Self: Sized,
    {
        match self {
            FractionMatrixEnum::Approx(m) => Ok(FractionMatrixEnum::Approx(m.invert()?)),
            FractionMatrixEnum::Exact(m) => Ok(FractionMatrixEnum::Exact(m.invert()?)),
            FractionMatrixEnum::CannotCombineExactAndApprox => {
                Err(anyhow!("cannot combine exact and approximate arithmetic"))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        fraction::{fraction_enum::FractionEnum, fraction_f64::FractionF64},
        matrix::{
            fraction_matrix_enum::FractionMatrixEnum, fraction_matrix_exact::FractionMatrixExact,
            fraction_matrix_f64::FractionMatrixF64, inversion::Inversion,
        },
    };

    #[test]
    fn inverse_f64() {
        let mut m: FractionMatrixF64 = vec![
            vec![1.into(), 0.into(), 0.into(), 0.into()],
            vec![0.into(), 1.into(), 0.into(), FractionF64::from((-3, 5))],
            vec![0.into(), FractionF64::from((-3, 4)), 1.into(), 0.into()],
            vec![0.into(), 0.into(), 0.into(), 1.into()],
        ]
        .try_into()
        .unwrap();

        let i: FractionMatrixF64 = vec![
            vec![1.into(), 0.into(), 0.into(), 0.into()],
            vec![0.into(), 1.into(), 0.into(), FractionF64::from((3, 5))],
            vec![
                0.into(),
                FractionF64::from((3, 4)),
                1.into(),
                FractionF64::from((9, 20)),
            ],
            vec![0.into(), 0.into(), 0.into(), 1.into()],
        ]
        .try_into()
        .unwrap();

        m = m.invert().unwrap();

        // println!("\t\tinverted matrix    {:?}", m);
        // println!("\t\tcorrect inverse    {:?}", i);

        assert_eq!(m, i);
    }

    #[test]
    fn inverse_biguint() {
        let mut m: FractionMatrixExact = vec![
            vec![1.into(), 0.into(), 0.into(), 0.into()],
            vec![0.into(), 1.into(), 0.into(), (-3, 5).into()],
            vec![0.into(), (-3, 4).into(), 1.into(), 0.into()],
            vec![0.into(), 0.into(), 0.into(), 1.into()],
        ]
        .try_into()
        .unwrap();

        let mut i: FractionMatrixExact = vec![
            vec![1.into(), 0.into(), 0.into(), 0.into()],
            vec![0.into(), 1.into(), 0.into(), (3, 5).into()],
            vec![0.into(), (3, 4).into(), 1.into(), (9, 20).into()],
            vec![0.into(), 0.into(), 0.into(), 1.into()],
        ]
        .try_into()
        .unwrap();

        m = m.invert().unwrap();

        // println!("\t\tinverted matrix    {:?}", m);
        // println!("\t\tcorrect inverse    {:?}", i);

        assert!(m.eq(&mut i));
    }

    #[test]
    fn inverse() {
        let mut m: FractionMatrixEnum = vec![
            vec![1.into(), 0.into(), 0.into(), 0.into()],
            vec![0.into(), 1.into(), 0.into(), FractionEnum::from((-3, 5))],
            vec![0.into(), FractionEnum::from((-3, 4)), 1.into(), 0.into()],
            vec![0.into(), 0.into(), 0.into(), 1.into()],
        ]
        .try_into()
        .unwrap();

        println!("matrix {}", m);

        let i = vec![
            vec![1.into(), 0.into(), 0.into(), 0.into()],
            vec![0.into(), 1.into(), 0.into(), FractionEnum::from((3, 5))],
            vec![
                0.into(),
                FractionEnum::from((3, 4)),
                1.into(),
                FractionEnum::from((9, 20)),
            ],
            vec![0.into(), 0.into(), 0.into(), 1.into()],
        ]
        .try_into()
        .unwrap();

        m = m.invert().unwrap();

        println!("inverted matrix {}", m);
        println!("correct inverse {}", i);

        assert_eq!(m, i);
    }
}
