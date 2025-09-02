use std::cmp::Ordering;

use crate::{
    ebi_number::{Signed, Zero},
    fraction::{
        fraction::EPSILON, fraction_enum::FractionEnum, fraction_exact::FractionExact,
        fraction_f64::FractionF64,
    },
};
use malachite::{
    Integer, Natural,
    base::num::{
        arithmetic::traits::{Abs, Sign},
        basic::traits::Zero as MZero,
    },
    rational::Rational,
};

impl Signed for FractionF64 {
    fn abs(self) -> Self {
        Self(self.0.abs())
    }

    fn is_positive(&self) -> bool {
        self.0 != 0f64 && self.0 > EPSILON
    }

    fn is_negative(&self) -> bool {
        self.0 != 0f64 && self.0 < -EPSILON
    }

    fn is_not_negative(&self) -> bool {
        self.0.is_not_negative()
    }

    fn is_not_positive(&self) -> bool {
        self.0.is_not_positive()
    }
}

impl Signed for FractionExact {
    fn abs(self) -> Self {
        Self(Signed::abs(self.0))
    }

    fn is_positive(&self) -> bool {
        self.0.is_positive()
    }

    fn is_negative(&self) -> bool {
        self.0.is_negative()
    }

    fn is_not_negative(&self) -> bool {
        !self.is_negative()
    }

    fn is_not_positive(&self) -> bool {
        !self.is_positive()
    }
}

impl Signed for FractionEnum {
    fn abs(self) -> Self {
        match self {
            FractionEnum::Exact(f) => FractionEnum::Exact(Abs::abs(f)),
            FractionEnum::Approx(f) => FractionEnum::Approx(f.abs()),
            FractionEnum::CannotCombineExactAndApprox => self.clone(),
        }
    }

    fn is_positive(&self) -> bool {
        match self {
            FractionEnum::Exact(f) => Signed::is_positive(f),
            FractionEnum::Approx(f) => Signed::is_positive(f),
            FractionEnum::CannotCombineExactAndApprox => false,
        }
    }

    fn is_negative(&self) -> bool {
        match self {
            FractionEnum::Exact(f) => Signed::is_negative(f),
            FractionEnum::Approx(f) => Signed::is_negative(f),
            FractionEnum::CannotCombineExactAndApprox => false,
        }
    }

    fn is_not_negative(&self) -> bool {
        match self {
            FractionEnum::Exact(_) => !self.is_negative(),
            FractionEnum::Approx(f) => f.is_not_negative(),
            FractionEnum::CannotCombineExactAndApprox => false,
        }
    }

    fn is_not_positive(&self) -> bool {
        match self {
            FractionEnum::Exact(_) => !self.is_positive(),
            FractionEnum::Approx(f) => f.is_not_positive(),
            FractionEnum::CannotCombineExactAndApprox => false,
        }
    }
}

impl Signed for Rational {
    fn abs(self) -> Self {
        Abs::abs(self)
    }

    fn is_positive(&self) -> bool {
        self > &Rational::ZERO
    }

    fn is_negative(&self) -> bool {
        self < &Rational::ZERO
    }
}

impl Signed for Integer {
    fn abs(self) -> Self {
        Abs::abs(self)
    }

    fn is_positive(&self) -> bool {
        self > &Integer::ZERO
    }

    fn is_negative(&self) -> bool {
        self < &Integer::ZERO
    }
}

impl Signed for Natural {
    fn abs(self) -> Self {
        self
    }

    fn is_positive(&self) -> bool {
        self > &Natural::ZERO
    }

    fn is_negative(&self) -> bool {
        false
    }
}

macro_rules! float {
    ($t: ident, $e: expr) => {
        impl Signed for $t {
            fn abs(self) -> Self {
                $t::abs(self)
            }

            fn is_positive(&self) -> bool {
                *self != 0.0 && self > &$e
            }

            fn is_negative(&self) -> bool {
                *self != 0.0 && self < &-$e
            }

            fn is_not_negative(&self) -> bool {
                self > &-$e
            }

            fn is_not_positive(&self) -> bool {
                self < &$e
            }
        }
    };
}

float!(f32, f32::EPSILON);
float!(f64, EPSILON);

macro_rules! ttype {
    ($t:ident) => {
        impl Signed for $t {
            fn abs(self) -> Self {
                self
            }

            fn is_positive(&self) -> bool {
                *self > 0
            }

            fn is_negative(&self) -> bool {
                false
            }
        }
    };
}

macro_rules! ttype_signed {
    ($t:ident) => {
        impl Signed for $t {
            fn abs(self) -> Self {
                $t::abs(self)
            }

            fn is_positive(&self) -> bool {
                self > &$t::zero()
            }

            fn is_negative(&self) -> bool {
                self < &$t::zero()
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
ttype_signed!(i128);
ttype_signed!(i64);
ttype_signed!(i32);
ttype_signed!(i16);
ttype_signed!(i8);

pub trait Numerator {
    fn signed_numerator(&self) -> Integer;
}

impl Numerator for Rational {
    fn signed_numerator(&self) -> Integer {
        if self.sign() == Ordering::Less {
            -self.numerator_ref()
        } else {
            self.to_numerator().into()
        }
    }
}
