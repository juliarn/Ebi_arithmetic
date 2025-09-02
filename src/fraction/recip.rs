use malachite::{base::num::arithmetic::traits::Reciprocal, rational::Rational};

use crate::{
    ebi_number::Recip,
    fraction::{
        fraction_enum::FractionEnum, fraction_exact::FractionExact, fraction_f64::FractionF64,
    },
};

impl Recip for FractionF64 {
    fn recip(self) -> Self {
        Self(self.0.recip())
    }
}

impl Recip for FractionExact {
    fn recip(self) -> Self {
        Self(self.0.recip())
    }
}

impl Recip for FractionEnum {
    fn recip(self) -> Self {
        match self {
            FractionEnum::Exact(f) => FractionEnum::Exact(f.recip()),
            FractionEnum::Approx(f) => FractionEnum::Approx(f.recip()),
            FractionEnum::CannotCombineExactAndApprox => FractionEnum::CannotCombineExactAndApprox,
        }
    }
}

impl Recip for f64 {
    fn recip(self) -> Self {
        1.0 / self
    }
}

impl Recip for Rational {
    fn recip(self) -> Self {
        Rational::reciprocal(self)
    }
}
