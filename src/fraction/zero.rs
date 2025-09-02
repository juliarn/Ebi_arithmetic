use malachite::{Integer, Natural, base::num::basic::traits::Zero as MZero, rational::Rational};

use crate::{
    ebi_number::{Signed, Zero},
    exact::is_exact_globally,
    fraction::{
        fraction::EPSILON, fraction_enum::FractionEnum, fraction_exact::FractionExact,
        fraction_f64::FractionF64,
    },
};

impl Zero for FractionF64 {
    fn zero() -> Self {
        Self(0.0)
    }

    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl Zero for FractionExact {
    fn zero() -> Self {
        Self(Rational::ZERO.clone())
    }

    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl Zero for FractionEnum {
    fn zero() -> Self {
        if is_exact_globally() {
            FractionEnum::Exact(Rational::ZERO)
        } else {
            FractionEnum::Approx(0.0)
        }
    }

    fn is_zero(&self) -> bool {
        match self {
            FractionEnum::Exact(f) => f.is_zero(),
            FractionEnum::Approx(f) => f.is_zero(),
            Self::CannotCombineExactAndApprox => false,
        }
    }

    fn set_zero(&mut self) {
        match self {
            FractionEnum::Exact(f) => *f = Rational::ZERO,
            FractionEnum::Approx(f) => *f = 0.0,
            FractionEnum::CannotCombineExactAndApprox => {}
        }
    }
}

impl Zero for Rational {
    fn zero() -> Self {
        Rational::ZERO.clone()
    }

    fn is_zero(&self) -> bool {
        self == &Rational::ZERO
    }
}

impl Zero for Natural {
    fn zero() -> Self {
        Natural::ZERO.clone()
    }

    fn is_zero(&self) -> bool {
        self == &Natural::ZERO
    }
}

impl Zero for Integer {
    fn zero() -> Self {
        Integer::ZERO.clone()
    }

    fn is_zero(&self) -> bool {
        self == &Integer::ZERO
    }
}

macro_rules! float {
    ($t: ident, $e: expr) => {
        impl Zero for $t {
            fn zero() -> Self {
                0.0
            }

            fn is_zero(&self) -> bool {
                <$t as Signed>::abs(*self) - &$e < 0.0
            }
        }
    };
}

float!(f32, f32::EPSILON);
float!(f64, EPSILON);

macro_rules! ttype {
    ($t:ident) => {
        impl Zero for $t {
            fn zero() -> Self {
                0
            }

            fn is_zero(&self) -> bool {
                *self == 0
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
