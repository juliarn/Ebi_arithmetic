use crate::fraction::{
    fraction_enum::FractionEnum, fraction_exact::FractionExact, fraction_f64::FractionF64,
};
use anyhow::{Result, anyhow};
use malachite::{
    Rational,
    base::{num::conversion::traits::RoundingFrom, rounding_modes::RoundingMode},
};

pub trait Approximate {
    /// Returns the f64 that is closest to the given number.
    /// Warning: do not convert a Fraction to an f64 and then back to a Fraction. Obviously, exactness is then lost.
    fn approximate(self) -> Result<f64>;
}

impl Approximate for FractionF64 {
    fn approximate(self) -> Result<f64> {
        self.0.approximate()
    }
}

impl Approximate for f64 {
    fn approximate(self) -> Result<f64> {
        Ok(self)
    }
}

impl Approximate for FractionExact {
    fn approximate(self) -> Result<f64> {
        self.0.approximate()
    }
}

impl Approximate for Rational {
    fn approximate(self) -> Result<f64> {
        Ok(f64::rounding_from(self, RoundingMode::Nearest).0)
    }
}

impl Approximate for FractionEnum {
    fn approximate(self) -> Result<f64> {
        match self {
            FractionEnum::Exact(rational) => Approximate::approximate(rational),
            FractionEnum::Approx(f) => Approximate::approximate(f),
            FractionEnum::CannotCombineExactAndApprox => {
                Err(anyhow!("cannot combine approximate and exact arithmetic"))
            }
        }
    }
}
