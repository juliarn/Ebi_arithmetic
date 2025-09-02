use malachite::{
    base::num::arithmetic::traits::{Ceiling, Floor},
    rational::Rational,
};

use crate::{
    ebi_number::Round,
    fraction::{
        fraction_enum::FractionEnum, fraction_exact::FractionExact, fraction_f64::FractionF64,
    },
};

impl Round for FractionF64 {
    fn floor(self) -> Self {
        FractionF64(self.0.floor())
    }

    fn ceil(self) -> Self {
        FractionF64(self.0.ceil())
    }
}

impl Round for FractionExact {
    fn floor(self) -> Self {
        Self(Round::floor(self.0))
    }

    fn ceil(self) -> Self {
        Self(Round::ceil(self.0))
    }
}

impl Round for FractionEnum {
    fn floor(self) -> Self {
        match self {
            Self::Exact(f) => Self::Exact(Round::floor(f)),
            Self::Approx(f) => Self::Approx(f.floor()),
            Self::CannotCombineExactAndApprox => Self::CannotCombineExactAndApprox,
        }
    }

    fn ceil(self) -> Self {
        match self {
            Self::Exact(f) => Self::Exact(f.ceil()),
            Self::Approx(f) => Self::Approx(f.ceil()),
            Self::CannotCombineExactAndApprox => Self::CannotCombineExactAndApprox,
        }
    }
}

impl Round for Rational {
    fn floor(self) -> Self {
        Floor::floor(self).into()
    }

    fn ceil(self) -> Self {
        Ceiling::ceiling(self).into()
    }
}

macro_rules! float {
    ($t: ident, $e: expr) => {
        impl Round for $t {
            fn floor(self) -> $t {
                $t::floor(self)
            }

            fn ceil(self) -> $t {
                $t::ceil(self)
            }
        }
    };
}

float!(f32, f32::EPSILON);
float!(f64, EPSILON);

macro_rules! ttype {
    ($t:ident) => {
        impl Round for $t {
            fn floor(self) -> Self {
                self
            }

            fn ceil(self) -> Self {
                self
            }
        }
    };
}

ttype!(usize);
ttype!(u128);
ttype!(u64);
ttype!(u32);
ttype!(u16);
ttype!(u8);
ttype!(i128);
ttype!(i64);
ttype!(i32);
ttype!(i16);
ttype!(i8);
