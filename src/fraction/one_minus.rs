use malachite::{Integer, base::num::basic::traits::One, rational::Rational};

use crate::{
    OneMinus,
    fraction::{
        fraction_enum::FractionEnum, fraction_exact::FractionExact, fraction_f64::FractionF64,
    },
};

impl OneMinus for FractionF64 {
    fn one_minus(self) -> Self {
        Self(1.0 - self.0)
    }
}

impl OneMinus for FractionExact {
    fn one_minus(self) -> Self {
        FractionExact(Rational::ONE - self.0)
    }
}

impl OneMinus for FractionEnum {
    fn one_minus(self) -> Self {
        match self {
            FractionEnum::Exact(f) => FractionEnum::Exact(f.one_minus()),
            FractionEnum::Approx(f) => FractionEnum::Approx(f.one_minus()),
            FractionEnum::CannotCombineExactAndApprox => FractionEnum::CannotCombineExactAndApprox,
        }
    }
}

impl OneMinus for Rational {
    fn one_minus(self) -> Self {
        Rational::ONE - self
    }
}

impl OneMinus for Integer {
    fn one_minus(self) -> Self {
        Integer::ONE - self
    }
}

macro_rules! one_minus_float {
    ($t:ident) => {
        impl OneMinus for $t {
            fn one_minus(self) -> Self {
                1.0 - self
            }
        }
    };
}

one_minus_float!(f64);
one_minus_float!(f32);

macro_rules! one_minus {
    ($t:ident) => {
        impl OneMinus for $t {
            fn one_minus(self) -> Self {
                1 - self
            }
        }
    };
}

one_minus!(i128);
one_minus!(i64);
one_minus!(i32);
one_minus!(i16);
one_minus!(i8);
