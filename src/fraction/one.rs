use malachite::{base::num::basic::traits::One as MOne, rational::Rational, Integer, Natural};

use crate::{
    ebi_number::One,
    exact::is_exact_globally,
    fraction::{
        fraction::EPSILON, fraction_enum::FractionEnum, fraction_exact::FractionExact,
        fraction_f64::FractionF64,
    },
};

impl One for FractionF64 {
    fn one() -> Self {
        Self(1.0)
    }

    fn is_one(&self) -> bool {
        self.0.is_one()
    }
}

impl One for FractionExact {
    fn one() -> Self {
        Self(Rational::ONE)
    }

    fn is_one(&self) -> bool {
        self.0.is_one()
    }
}

impl One for FractionEnum {
    fn one() -> Self {
        if is_exact_globally() {
            FractionEnum::Exact(Rational::ONE)
        } else {
            FractionEnum::Approx(1.0)
        }
    }

    fn is_one(&self) -> bool {
        match self {
            FractionEnum::Exact(f) => f.is_one(),
            FractionEnum::Approx(f) => f.is_one(),
            Self::CannotCombineExactAndApprox => false,
        }
    }

    fn set_one(&mut self) {
        match self {
            FractionEnum::Exact(f) => f.set_one(),
            FractionEnum::Approx(f) => *f = 1.0,
            FractionEnum::CannotCombineExactAndApprox => {}
        }
    }
}

impl One for Rational {
    fn one() -> Self {
        Rational::ONE.clone()
    }

    fn is_one(&self) -> bool {
        self == &Rational::ONE
    }
}

impl One for Natural {
    fn one() -> Self {
        Natural::ONE.clone()
    }

    fn is_one(&self) -> bool {
        self == &Natural::ONE
    }
}

impl One for Integer {
    fn one() -> Self {
        Integer::ONE.clone()
    }

    fn is_one(&self) -> bool {
        self == &Integer::ONE
    }
}

macro_rules! float {
    ($t: ident, $e: expr) => {
        impl One for $t {
            fn one() -> Self {
                1.0
            }

            fn is_one(&self) -> bool {
                (self - 1.0).abs() - &$e < 0.0
            }
        }
    };
}

float!(f32, f32::EPSILON);
float!(f64, EPSILON);

macro_rules! ttype {
    ($t:ident) => {
        impl One for $t {
            fn one() -> Self {
                1
            }

            fn is_one(&self) -> bool {
                self == &1
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
