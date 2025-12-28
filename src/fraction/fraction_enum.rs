use crate::{
    ebi_number::Zero,
    exact::is_exact_globally,
    fraction::{fraction::EPSILON, fraction_exact::FractionExact},
};
use anyhow::{Error, anyhow};
use malachite::{
    base::{num::conversion::traits::RoundingFrom, rounding_modes::RoundingMode::Nearest},
    rational::Rational,
};
use std::{
    borrow::Borrow,
    cmp::Ordering,
    f64,
    hash::Hash,
    iter::Sum,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    str::FromStr,
    sync::Arc,
};

#[derive(Clone)]
pub enum FractionEnum {
    Exact(Rational),
    Approx(f64),
    CannotCombineExactAndApprox,
}

impl FractionEnum {
    /**
     * Returns whether the two given fractions are either both exact or both approximate
     */
    pub(crate) fn matches(&self, rhs: &Self) -> bool {
        match (self, rhs) {
            (FractionEnum::Exact(_), FractionEnum::Exact(_)) => true,
            (FractionEnum::Approx(_), FractionEnum::Approx(_)) => true,
            _ => false,
        }
    }
}

impl Default for FractionEnum {
    fn default() -> Self {
        Self::zero()
    }
}

impl FromStr for FractionEnum {
    type Err = Error;

    fn from_str(s: &str) -> std::prelude::v1::Result<Self, Self::Err> {
        if is_exact_globally() {
            match FractionExact::from_str(s) {
                Ok(x) => Ok(FractionEnum::Exact(x.0)),
                Err(_) => Err(anyhow!("{} is not a fraction", s)),
            }
        } else {
            if let Ok(float) = f64::from_str(s) {
                Ok(FractionEnum::Approx(float))
            } else {
                match Rational::from_str(s) {
                    Ok(f) => Ok(FractionEnum::Approx(f64::rounding_from(f, Nearest).0)),
                    Err(_) => Err(anyhow!("{} is not an approximate fraction", s)),
                }
            }
        }
    }
}

impl From<&FractionEnum> for FractionEnum {
    fn from(value: &FractionEnum) -> Self {
        match value {
            FractionEnum::Exact(_) => value.clone(),
            FractionEnum::Approx(_) => value.clone(),
            FractionEnum::CannotCombineExactAndApprox => value.clone(),
        }
    }
}

impl From<Arc<FractionEnum>> for FractionEnum {
    fn from(value: Arc<FractionEnum>) -> Self {
        match value.as_ref() {
            FractionEnum::Exact(f) => FractionEnum::Exact(f.clone()),
            FractionEnum::Approx(f) => FractionEnum::Approx(f.clone()),
            FractionEnum::CannotCombineExactAndApprox => FractionEnum::CannotCombineExactAndApprox,
        }
    }
}

impl From<&Arc<FractionEnum>> for FractionEnum {
    fn from(value: &Arc<FractionEnum>) -> Self {
        match value.as_ref() {
            FractionEnum::Exact(f) => FractionEnum::Exact(f.clone()),
            FractionEnum::Approx(f) => FractionEnum::Approx(f.clone()),
            FractionEnum::CannotCombineExactAndApprox => FractionEnum::CannotCombineExactAndApprox,
        }
    }
}

#[macro_export]
/// Convenience short-hand macro to create fractions.
macro_rules! f_en {
    ($e: expr) => {
        FractionEnum::from($e)
    };

    ($e: expr, $f: expr) => {
        FractionEnum::from(($e, $f))
    };
}
pub use f_en;

#[macro_export]
/// Convenience short-hand macro to create a fraction representing zero.
macro_rules! f0_en {
    () => {
        FractionEnum::zero()
    };
}
pub use f0_en;

#[macro_export]
/// Convenience short-hand macro to create a fraction representing one.
macro_rules! f1_en {
    () => {
        FractionEnum::one()
    };
}
pub use f1_en;

impl std::fmt::Display for FractionEnum {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FractionEnum::Exact(fr) => std::fmt::Display::fmt(&fr, f),
            FractionEnum::Approx(fr) => std::fmt::Display::fmt(&fr, f),
            FractionEnum::CannotCombineExactAndApprox => {
                write!(f, "cannot combine exact and approximate arithmatic")
            }
        }
    }
}

impl std::fmt::Debug for FractionEnum {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Exact(arg0) => f.debug_tuple("Exact ").field(arg0).finish(),
            Self::Approx(arg0) => f.debug_tuple("Approx ").field(arg0).finish(),
            Self::CannotCombineExactAndApprox => {
                write!(f, "cannot combine exact and approximate arithmatic")
            }
        }
    }
}

impl Add<&FractionEnum> for &FractionEnum {
    type Output = FractionEnum;

    fn add(self, rhs: &FractionEnum) -> Self::Output {
        match (self, rhs) {
            (FractionEnum::Exact(x), FractionEnum::Exact(y)) => FractionEnum::Exact(x.add(y)),
            (FractionEnum::Approx(x), FractionEnum::Approx(y)) => FractionEnum::Approx(x.add(y)),
            _ => FractionEnum::CannotCombineExactAndApprox,
        }
    }
}

impl Add<FractionEnum> for FractionEnum {
    type Output = FractionEnum;

    fn add(self, rhs: FractionEnum) -> Self::Output {
        match (self, rhs) {
            (FractionEnum::Exact(x), FractionEnum::Exact(y)) => FractionEnum::Exact(x.add(y)),
            (FractionEnum::Approx(x), FractionEnum::Approx(y)) => FractionEnum::Approx(x.add(y)),
            _ => FractionEnum::CannotCombineExactAndApprox,
        }
    }
}

impl<T> AddAssign<T> for FractionEnum
where
    T: Borrow<FractionEnum>,
{
    fn add_assign(&mut self, rhs: T) {
        let rhs = rhs.borrow();

        if self.matches(&rhs) {
            match (self, rhs) {
                (FractionEnum::Exact(x), FractionEnum::Exact(y)) => x.add_assign(y),
                (FractionEnum::Approx(x), FractionEnum::Approx(y)) => x.add_assign(y),
                _ => {}
            };
        } else {
            *self = FractionEnum::CannotCombineExactAndApprox
        }
    }
}

impl AddAssign<&Arc<FractionEnum>> for FractionEnum {
    fn add_assign(&mut self, rhs: &Arc<FractionEnum>) {
        let rhs = rhs.borrow();

        if self.matches(&rhs) {
            match (self, rhs) {
                (FractionEnum::Exact(x), FractionEnum::Exact(y)) => x.add_assign(y),
                (FractionEnum::Approx(x), FractionEnum::Approx(y)) => x.add_assign(y),
                _ => {}
            };
        } else {
            *self = FractionEnum::CannotCombineExactAndApprox
        }
    }
}

impl Sub<&FractionEnum> for &FractionEnum {
    type Output = FractionEnum;

    fn sub(self, rhs: &FractionEnum) -> Self::Output {
        match (self, rhs) {
            (FractionEnum::Exact(x), FractionEnum::Exact(y)) => FractionEnum::Exact(x.sub(y)),
            (FractionEnum::Approx(x), FractionEnum::Approx(y)) => FractionEnum::Approx(x.sub(y)),
            _ => FractionEnum::CannotCombineExactAndApprox,
        }
    }
}

impl Sub<FractionEnum> for FractionEnum {
    type Output = FractionEnum;

    fn sub(self, rhs: FractionEnum) -> Self::Output {
        match (self, rhs) {
            (FractionEnum::Exact(x), FractionEnum::Exact(y)) => FractionEnum::Exact(x.sub(y)),
            (FractionEnum::Approx(x), FractionEnum::Approx(y)) => FractionEnum::Approx(x.sub(y)),
            _ => FractionEnum::CannotCombineExactAndApprox,
        }
    }
}

impl<T> SubAssign<T> for FractionEnum
where
    T: Borrow<FractionEnum>,
{
    fn sub_assign(&mut self, rhs: T) {
        let rhs = rhs.borrow();
        if self.matches(&rhs) {
            match (self, rhs) {
                (FractionEnum::Exact(x), FractionEnum::Exact(y)) => x.sub_assign(y),
                (FractionEnum::Approx(x), FractionEnum::Approx(y)) => x.sub_assign(y),
                _ => {}
            }
        } else {
            *self = FractionEnum::CannotCombineExactAndApprox;
        }
    }
}

impl Mul<&FractionEnum> for &FractionEnum {
    type Output = FractionEnum;

    fn mul(self, rhs: &FractionEnum) -> Self::Output {
        match (self, rhs) {
            (FractionEnum::Exact(x), FractionEnum::Exact(y)) => FractionEnum::Exact(x.mul(y)),
            (FractionEnum::Approx(x), FractionEnum::Approx(y)) => FractionEnum::Approx(x.mul(y)),
            _ => FractionEnum::CannotCombineExactAndApprox,
        }
    }
}

impl Mul<FractionEnum> for FractionEnum {
    type Output = FractionEnum;

    fn mul(self, rhs: FractionEnum) -> Self::Output {
        match (self, rhs) {
            (FractionEnum::Exact(x), FractionEnum::Exact(y)) => FractionEnum::Exact(x.mul(y)),
            (FractionEnum::Approx(x), FractionEnum::Approx(y)) => FractionEnum::Approx(x.mul(y)),
            _ => FractionEnum::CannotCombineExactAndApprox,
        }
    }
}

impl<T> MulAssign<T> for FractionEnum
where
    T: Borrow<FractionEnum>,
{
    fn mul_assign(&mut self, rhs: T) {
        let rhs = rhs.borrow();
        if self.matches(&rhs) {
            match (self, rhs) {
                (FractionEnum::Exact(x), FractionEnum::Exact(y)) => x.mul_assign(y),
                (FractionEnum::Approx(x), FractionEnum::Approx(y)) => x.mul_assign(y),
                _ => {}
            }
        } else {
            *self = FractionEnum::CannotCombineExactAndApprox
        }
    }
}

impl Div<&FractionEnum> for &FractionEnum {
    type Output = FractionEnum;

    fn div(self, rhs: &FractionEnum) -> Self::Output {
        match (self, rhs) {
            (FractionEnum::Exact(x), FractionEnum::Exact(y)) => FractionEnum::Exact(x.div(y)),
            (FractionEnum::Approx(x), FractionEnum::Approx(y)) => FractionEnum::Approx(x.div(y)),
            _ => FractionEnum::CannotCombineExactAndApprox,
        }
    }
}

impl Div<FractionEnum> for FractionEnum {
    type Output = FractionEnum;

    fn div(self, rhs: FractionEnum) -> Self::Output {
        match (self, rhs) {
            (FractionEnum::Exact(x), FractionEnum::Exact(y)) => FractionEnum::Exact(x.div(y)),
            (FractionEnum::Approx(x), FractionEnum::Approx(y)) => FractionEnum::Approx(x.div(y)),
            _ => FractionEnum::CannotCombineExactAndApprox,
        }
    }
}

impl<T> DivAssign<T> for FractionEnum
where
    T: Borrow<FractionEnum>,
{
    fn div_assign(&mut self, rhs: T) {
        let rhs = rhs.borrow();
        if self.matches(&rhs) {
            match (self, rhs) {
                (FractionEnum::Exact(x), FractionEnum::Exact(y)) => x.div_assign(y),
                (FractionEnum::Approx(x), FractionEnum::Approx(y)) => x.div_assign(y),
                _ => {}
            }
        } else {
            *self = FractionEnum::CannotCombineExactAndApprox
        }
    }
}

impl Neg for FractionEnum {
    type Output = FractionEnum;

    fn neg(self) -> Self::Output {
        match self {
            FractionEnum::Exact(f) => FractionEnum::Exact(f.neg()),
            FractionEnum::Approx(f) => FractionEnum::Approx(f.neg()),
            Self::CannotCombineExactAndApprox => self.clone(),
        }
    }
}

impl<'a> Neg for &'a FractionEnum {
    type Output = FractionEnum;

    fn neg(self) -> Self::Output {
        match self {
            FractionEnum::Exact(f) => FractionEnum::Exact(f.neg()),
            FractionEnum::Approx(f) => FractionEnum::Approx(f.neg()),
            FractionEnum::CannotCombineExactAndApprox => self.clone(),
        }
    }
}

impl PartialEq for FractionEnum {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Exact(l0), Self::Exact(r0)) => l0 == r0,
            (Self::Approx(l0), Self::Approx(r0)) => l0 - EPSILON <= *r0 && *r0 <= l0 + EPSILON,
            _ => false,
        }
    }
}

impl Eq for FractionEnum {}

impl PartialOrd for FractionEnum {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (FractionEnum::Exact(x), FractionEnum::Exact(y)) => x.partial_cmp(y),
            (FractionEnum::Approx(x), FractionEnum::Approx(y)) => x.partial_cmp(y),
            _ => None,
        }
    }
}

impl Ord for FractionEnum {
    /**
     * Note that exact and approximate should not be compared.
     */
    fn cmp(&self, other: &Self) -> Ordering {
        if !self.matches(other) {
            panic!("cannot compare exact and inexact arithmethic");
        }
        match (self, other) {
            (FractionEnum::Exact(x), FractionEnum::Exact(y)) => x.cmp(y),
            (FractionEnum::Approx(x), FractionEnum::Approx(y)) => {
                if x.is_nan() && y.is_nan() {
                    Ordering::Equal
                } else if x.is_nan() {
                    Ordering::Less
                } else if y.is_nan() {
                    Ordering::Greater
                } else if x == &f64::INFINITY {
                    if y == &f64::INFINITY {
                        Ordering::Equal
                    } else {
                        Ordering::Greater
                    }
                } else if y == &f64::INFINITY {
                    Ordering::Less
                } else if x == &f64::NEG_INFINITY {
                    if y == &f64::NEG_INFINITY {
                        Ordering::Equal
                    } else {
                        Ordering::Less
                    }
                } else if y == &f64::NEG_INFINITY {
                    Ordering::Greater
                } else {
                    x.partial_cmp(y).unwrap()
                }
            }
            (FractionEnum::Exact(_), FractionEnum::Approx(_)) => Ordering::Greater,
            (FractionEnum::Exact(_), FractionEnum::CannotCombineExactAndApprox) => {
                Ordering::Greater
            }
            (FractionEnum::Approx(_), FractionEnum::Exact(_)) => Ordering::Less,
            (FractionEnum::Approx(_), FractionEnum::CannotCombineExactAndApprox) => {
                Ordering::Greater
            }
            (FractionEnum::CannotCombineExactAndApprox, FractionEnum::Exact(_)) => Ordering::Less,
            (FractionEnum::CannotCombineExactAndApprox, FractionEnum::Approx(_)) => Ordering::Less,
            (
                FractionEnum::CannotCombineExactAndApprox,
                FractionEnum::CannotCombineExactAndApprox,
            ) => Ordering::Less,
        }
    }
}

impl Hash for FractionEnum {
    /**
     * For good reasons, Rust does not support hashing of doubles. However, we need it to store distributions in a hashmap.
     * Approximate arithmetic is discouraged
     */
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            FractionEnum::Exact(f) => f.hash(state),
            FractionEnum::Approx(f) => f64::to_bits(*f).hash(state),
            Self::CannotCombineExactAndApprox => "cceaa".hash(state),
        }
    }
}

impl Sum for FractionEnum {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |sum, f| &sum + &f)
    }
}

impl<'a> Sum<&'a FractionEnum> for FractionEnum {
    fn sum<I: Iterator<Item = &'a FractionEnum>>(iter: I) -> Self {
        iter.fold(FractionEnum::zero(), |sum, f| &sum + f)
    }
}

//======================== froms ========================//

macro_rules! from_1 {
    ($t:ident) => {
        impl From<$t> for FractionEnum {
            fn from(value: $t) -> Self {
                if is_exact_globally() {
                    FractionEnum::Exact(Rational::from(value))
                } else {
                    FractionEnum::Approx(value as f64)
                }
            }
        }
    };
}

macro_rules! from_2 {
    ($t:ident,$tt:ident) => {
        impl From<($t, $tt)> for FractionEnum {
            fn from(value: ($t, $tt)) -> Self {
                if is_exact_globally() {
                    FractionEnum::Exact(Rational::from(value.0) / Rational::from(value.1))
                } else {
                    FractionEnum::Approx(value.0 as f64 / value.1 as f64)
                }
            }
        }
    };
}

macro_rules! from_3 {
    ($t:ident) => {
        from_1!($t);
        from_2!($t, usize);
        from_2!($t, u128);
        from_2!($t, u64);
        from_2!($t, u32);
        from_2!($t, u16);
        from_2!($t, u8);
        from_2!($t, i128);
        from_2!($t, i64);
        from_2!($t, i32);
        from_2!($t, i16);
        from_2!($t, i8);
    };
}

from_3!(usize);
from_3!(u128);
from_3!(u64);
from_3!(u32);
from_3!(u16);
from_3!(u8);
from_3!(i128);
from_3!(i64);
from_3!(i32);
from_3!(i16);
from_3!(i8);

//======================== operators ========================//

macro_rules! add {
    ($t:ident) => {
        impl<'a> Add<$t> for &'a FractionEnum {
            type Output = FractionEnum;

            fn add(self, rhs: $t) -> Self::Output {
                let rhs = rhs.into();
                match (self, rhs) {
                    (FractionEnum::Exact(x), FractionEnum::Exact(y)) => {
                        FractionEnum::Exact(x.add(y))
                    }
                    (FractionEnum::Approx(x), FractionEnum::Approx(y)) => {
                        FractionEnum::Approx(x.add(y))
                    }
                    _ => FractionEnum::CannotCombineExactAndApprox,
                }
            }
        }
    };
}

macro_rules! add_assign {
    ($t:ident) => {
        impl AddAssign<$t> for FractionEnum {
            fn add_assign(&mut self, rhs: $t) {
                let rhs = rhs.into();
                if self.matches(&rhs) {
                    match (self, rhs) {
                        (FractionEnum::Exact(x), FractionEnum::Exact(y)) => x.add_assign(y),
                        (FractionEnum::Approx(x), FractionEnum::Approx(y)) => x.add_assign(y),
                        _ => {}
                    };
                } else {
                    *self = FractionEnum::CannotCombineExactAndApprox
                }
            }
        }
    };
}

macro_rules! sub {
    ($t:ident) => {
        impl<'a> Sub<$t> for &'a FractionEnum {
            type Output = FractionEnum;

            fn sub(self, rhs: $t) -> Self::Output {
                let rhs = rhs.into();
                match (self, rhs) {
                    (FractionEnum::Exact(x), FractionEnum::Exact(y)) => {
                        FractionEnum::Exact(x.sub(y))
                    }
                    (FractionEnum::Approx(x), FractionEnum::Approx(y)) => {
                        FractionEnum::Approx(x.sub(y))
                    }
                    _ => FractionEnum::CannotCombineExactAndApprox,
                }
            }
        }
    };
}

macro_rules! sub_assign {
    ($t:ident) => {
        impl SubAssign<$t> for FractionEnum {
            fn sub_assign(&mut self, rhs: $t) {
                let rhs = rhs.into();
                if self.matches(&rhs) {
                    match (self, rhs) {
                        (FractionEnum::Exact(x), FractionEnum::Exact(y)) => x.sub_assign(y),
                        (FractionEnum::Approx(x), FractionEnum::Approx(y)) => x.sub_assign(y),
                        _ => {}
                    };
                } else {
                    *self = FractionEnum::CannotCombineExactAndApprox
                }
            }
        }
    };
}

macro_rules! mul {
    ($t:ident) => {
        impl<'a> Mul<$t> for &'a FractionEnum {
            type Output = FractionEnum;

            fn mul(self, rhs: $t) -> Self::Output {
                let rhs = rhs.into();
                match (self, rhs) {
                    (FractionEnum::Exact(x), FractionEnum::Exact(y)) => {
                        FractionEnum::Exact(x.mul(y))
                    }
                    (FractionEnum::Approx(x), FractionEnum::Approx(y)) => {
                        FractionEnum::Approx(x.mul(y))
                    }
                    _ => FractionEnum::CannotCombineExactAndApprox,
                }
            }
        }
    };
}

macro_rules! mul_assign {
    ($t:ident) => {
        impl MulAssign<$t> for FractionEnum {
            fn mul_assign(&mut self, rhs: $t) {
                let rhs = rhs.into();
                if self.matches(&rhs) {
                    match (self, rhs) {
                        (FractionEnum::Exact(x), FractionEnum::Exact(y)) => x.mul_assign(y),
                        (FractionEnum::Approx(x), FractionEnum::Approx(y)) => x.mul_assign(y),
                        _ => {}
                    };
                } else {
                    *self = FractionEnum::CannotCombineExactAndApprox
                }
            }
        }
    };
}

macro_rules! div {
    ($t:ident) => {
        impl<'a> Div<$t> for &'a FractionEnum {
            type Output = FractionEnum;

            fn div(self, rhs: $t) -> Self::Output {
                let rhs = rhs.into();
                match (self, rhs) {
                    (FractionEnum::Exact(x), FractionEnum::Exact(y)) => {
                        FractionEnum::Exact(x.div(y))
                    }
                    (FractionEnum::Approx(x), FractionEnum::Approx(y)) => {
                        FractionEnum::Approx(x.div(y))
                    }
                    _ => FractionEnum::CannotCombineExactAndApprox,
                }
            }
        }
    };
}

macro_rules! div_assign {
    ($t:ident) => {
        impl DivAssign<$t> for FractionEnum {
            fn div_assign(&mut self, rhs: $t) {
                let rhs = rhs.into();
                if self.matches(&rhs) {
                    match (self, rhs) {
                        (FractionEnum::Exact(x), FractionEnum::Exact(y)) => x.div_assign(y),
                        (FractionEnum::Approx(x), FractionEnum::Approx(y)) => x.div_assign(y),
                        _ => {}
                    };
                } else {
                    *self = FractionEnum::CannotCombineExactAndApprox
                }
            }
        }
    };
}

macro_rules! ttype {
    ($t:ident) => {
        add!($t);
        add_assign!($t);
        sub!($t);
        sub_assign!($t);
        mul!($t);
        mul_assign!($t);
        div!($t);
        div_assign!($t);
    };
}

macro_rules! ttype_signed {
    ($t:ident) => {
        add!($t);
        add_assign!($t);
        sub!($t);
        sub_assign!($t);
        mul!($t);
        mul_assign!($t);
        div!($t);
        div_assign!($t);
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

#[cfg(test)]
mod tests {
    use crate::{
        ebi_number::{One, Signed},
        fraction::fraction_enum::FractionEnum,
    };
    use std::ops::Neg;

    #[test]
    fn fraction_neg() {
        let one = FractionEnum::one();
        assert!(one.is_positive());
        let one = one.neg();
        assert!(one.is_negative());
    }

    #[test]
    fn fraction_parse() {
        let x = "0.2".to_owned();
        let f: FractionEnum = x.parse().unwrap();
        assert_eq!(f, FractionEnum::from((1, 5)));

        assert_eq!("1".parse::<FractionEnum>().unwrap(), FractionEnum::one());
        assert_eq!("-1".parse::<FractionEnum>().unwrap(), -FractionEnum::one());

        assert_eq!("1.00".parse::<FractionEnum>().unwrap(), FractionEnum::one());
        assert_eq!(
            "-1.00".parse::<FractionEnum>().unwrap(),
            -FractionEnum::one()
        );

        assert_eq!(
            "1/5".parse::<FractionEnum>().unwrap(),
            FractionEnum::from((1, 5))
        );
        assert_eq!(
            "-1/5".parse::<FractionEnum>().unwrap(),
            -FractionEnum::from((1, 5))
        );

        assert_eq!(
            ".2".parse::<FractionEnum>().unwrap(),
            FractionEnum::from((1, 5))
        );
        assert_eq!(
            "-.2".parse::<FractionEnum>().unwrap(),
            -FractionEnum::from((1, 5))
        );
    }
}
