use core::f64;
use std::{
    borrow::Borrow,
    cmp::Ordering,
    fmt::Display,
    hash::Hash,
    iter::Sum,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    str::FromStr,
    sync::Arc,
};

use anyhow::{Error, anyhow};
use malachite::{
    base::{num::conversion::traits::RoundingFrom, rounding_modes::RoundingMode::Nearest},
    rational::Rational,
};

use crate::{ebi_number::Zero, fraction::fraction::EPSILON};

#[derive(Debug, Clone, Copy)]
pub struct FractionF64(pub(crate) f64);

impl Default for FractionF64 {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl PartialEq for FractionF64 {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (FractionF64(l0), FractionF64(r0)) => l0 - EPSILON <= *r0 && *r0 <= l0 + EPSILON,
        }
    }
}

impl Display for FractionF64 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl From<f64> for FractionF64 {
    fn from(value: f64) -> Self {
        Self(value)
    }
}

impl From<&FractionF64> for FractionF64 {
    fn from(value: &FractionF64) -> Self {
        value.clone()
    }
}

impl From<Arc<FractionF64>> for FractionF64 {
    fn from(value: Arc<FractionF64>) -> Self {
        value.as_ref().clone()
    }
}

impl From<&Arc<FractionF64>> for FractionF64 {
    fn from(value: &Arc<FractionF64>) -> Self {
        value.as_ref().clone()
    }
}

impl FromStr for FractionF64 {
    type Err = Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        if let Ok(float) = f64::from_str(s) {
            Ok(Self(float))
        } else {
            match Rational::from_str(s) {
                Ok(f) => Ok(Self(f64::rounding_from(f, Nearest).0)),
                Err(_) => Err(anyhow!("{} is not an approximate fraction", s)),
            }
        }
    }
}

#[macro_export]
/// Convenience short-hand macro to create fractions.
macro_rules! f_a {
    ($e: expr) => {
        FractionF64::from($e)
    };

    ($e: expr, $f: expr) => {
        FractionF64::from(($e, $f))
    };
}
pub use f_a;

#[macro_export]
/// Convenience short-hand macro to create a fraction representing zero.
macro_rules! f0_a {
    () => {
        FractionF64::zero()
    };
}
pub use f0_a;

#[macro_export]
/// Convenience short-hand macro to create a fraction representing one.
macro_rules! f1_a {
    () => {
        FractionF64::one()
    };
}
pub use f1_a;

impl Eq for FractionF64 {}

impl PartialOrd for FractionF64 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl Hash for FractionF64 {
    /**
     * For good reasons, Rust does not support hashing of doubles. However, we need it to store distributions in a hashmap.
     * Approximate arithmetic is discouraged
     */
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        f64::to_bits(self.0).hash(state)
    }
}

impl Ord for FractionF64 {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.0.is_nan() && other.0.is_nan() {
            Ordering::Equal
        } else if self.0.is_nan() {
            Ordering::Less
        } else if other.0.is_nan() {
            Ordering::Greater
        } else if self.0 == f64::INFINITY {
            if other.0 == f64::INFINITY {
                Ordering::Equal
            } else {
                Ordering::Greater
            }
        } else if other.0 == f64::INFINITY {
            Ordering::Less
        } else if self.0 == f64::NEG_INFINITY {
            if other.0 == f64::NEG_INFINITY {
                Ordering::Equal
            } else {
                Ordering::Less
            }
        } else if other.0 == f64::NEG_INFINITY {
            Ordering::Greater
        } else {
            self.0.partial_cmp(&other.0).unwrap()
        }
    }
}

impl Sum for FractionF64 {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |sum, f| &sum + &f)
    }
}

impl<'a> Sum<&'a FractionF64> for FractionF64 {
    fn sum<I: Iterator<Item = &'a FractionF64>>(iter: I) -> Self {
        iter.fold(FractionF64::zero(), |sum, f| &sum + f)
    }
}

impl Neg for FractionF64 {
    type Output = FractionF64;

    fn neg(self) -> Self::Output {
        Self(self.0.neg())
    }
}

impl<'a> Neg for &'a FractionF64 {
    type Output = FractionF64;

    fn neg(self) -> Self::Output {
        FractionF64(self.0.neg())
    }
}

impl Add<&FractionF64> for &FractionF64 {
    type Output = FractionF64;

    fn add(self, rhs: &FractionF64) -> Self::Output {
        FractionF64(self.0.add(rhs.0))
    }
}

impl Add<FractionF64> for FractionF64 {
    type Output = FractionF64;

    fn add(self, rhs: FractionF64) -> Self::Output {
        FractionF64(self.0.add(rhs.0))
    }
}

impl<T> AddAssign<T> for FractionF64
where
    T: Borrow<FractionF64>,
{
    fn add_assign(&mut self, rhs: T) {
        let rhs = rhs.borrow();
        self.0.add_assign(rhs.0)
    }
}

impl Sub<&FractionF64> for &FractionF64 {
    type Output = FractionF64;

    fn sub(self, rhs: &FractionF64) -> Self::Output {
        FractionF64(self.0.sub(rhs.0))
    }
}

impl Sub<FractionF64> for FractionF64 {
    type Output = FractionF64;

    fn sub(self, rhs: FractionF64) -> Self::Output {
        FractionF64(self.0.sub(rhs.0))
    }
}

impl<T> SubAssign<T> for FractionF64
where
    T: Borrow<FractionF64>,
{
    fn sub_assign(&mut self, rhs: T) {
        let rhs = rhs.borrow();
        self.0.sub_assign(rhs.0)
    }
}

impl Mul<&FractionF64> for &FractionF64 {
    type Output = FractionF64;

    fn mul(self, rhs: &FractionF64) -> Self::Output {
        FractionF64(self.0.mul(rhs.0))
    }
}

impl Mul<FractionF64> for FractionF64 {
    type Output = FractionF64;

    fn mul(self, rhs: FractionF64) -> Self::Output {
        FractionF64(self.0.mul(rhs.0))
    }
}

impl<T> MulAssign<T> for FractionF64
where
    T: Borrow<FractionF64>,
{
    fn mul_assign(&mut self, rhs: T) {
        let rhs = rhs.borrow();
        self.0.mul_assign(rhs.0)
    }
}

impl Div<&FractionF64> for &FractionF64 {
    type Output = FractionF64;

    fn div(self, rhs: &FractionF64) -> Self::Output {
        FractionF64(self.0.div(rhs.0))
    }
}

impl Div<FractionF64> for FractionF64 {
    type Output = FractionF64;

    fn div(self, rhs: FractionF64) -> Self::Output {
        FractionF64(self.0.div(rhs.0))
    }
}

impl<T> DivAssign<T> for FractionF64
where
    T: Borrow<FractionF64>,
{
    fn div_assign(&mut self, rhs: T) {
        let rhs = rhs.borrow();
        self.0.div_assign(rhs.0)
    }
}

//======================== primitive types ========================//

impl Mul<f64> for FractionF64 {
    type Output = FractionF64;

    fn mul(self, rhs: f64) -> Self::Output {
        Self(self.0 * rhs)
    }
}

impl Div<f64> for FractionF64 {
    type Output = FractionF64;

    fn div(self, rhs: f64) -> Self::Output {
        Self(self.0 / rhs)
    }
}

impl Add<f64> for FractionF64 {
    type Output = FractionF64;

    fn add(self, rhs: f64) -> Self::Output {
        Self(self.0 + rhs)
    }
}

impl Sub<f64> for FractionF64 {
    type Output = FractionF64;

    fn sub(self, rhs: f64) -> Self::Output {
        Self(self.0 - rhs)
    }
}

macro_rules! from {
    ($t:ident) => {
        impl From<$t> for FractionF64 {
            fn from(value: $t) -> Self {
                Self(value as f64)
            }
        }
    };
}

macro_rules! from_signed {
    ($t:ident) => {
        impl From<$t> for FractionF64 {
            fn from(value: $t) -> Self {
                Self(value as f64)
            }
        }
    };
}

macro_rules! from_tuple_u_u {
    ($t:ident,$tt:ident) => {
        impl From<($t, $tt)> for FractionF64 {
            fn from(value: ($t, $tt)) -> Self {
                Self(value.0 as f64 / value.1 as f64)
            }
        }
    };
}

macro_rules! from_tuple_u_i {
    ($t:ident,$tt:ident) => {
        impl From<($t, $tt)> for FractionF64 {
            fn from(value: ($t, $tt)) -> Self {
                Self(value.0 as f64 / value.1 as f64)
            }
        }
    };
}

macro_rules! from_tuple_i_u {
    ($t:ident,$tt:ident) => {
        impl From<($t, $tt)> for FractionF64 {
            fn from(value: ($t, $tt)) -> Self {
                Self(value.0 as f64 / value.1 as f64)
            }
        }
    };
}

macro_rules! from_tuple_i_i {
    ($t:ident,$tt:ident) => {
        impl From<($t, $tt)> for FractionF64 {
            fn from(value: ($t, $tt)) -> Self {
                Self(value.0 as f64 / value.1 as f64)
            }
        }
    };
}

macro_rules! add {
    ($t:ident) => {
        impl<'a> Add<$t> for &'a FractionF64 {
            type Output = FractionF64;

            fn add(self, rhs: $t) -> Self::Output {
                let rhs: FractionF64 = rhs.into();
                self.add(&rhs)
            }
        }
    };
}

macro_rules! add_assign {
    ($t:ident) => {
        impl AddAssign<$t> for FractionF64 {
            fn add_assign(&mut self, rhs: $t) {
                let rhs: FractionF64 = rhs.into();
                self.add_assign(rhs)
            }
        }
    };
}

macro_rules! sub {
    ($t:ident) => {
        impl<'a> Sub<$t> for &'a FractionF64 {
            type Output = FractionF64;

            fn sub(self, rhs: $t) -> Self::Output {
                let rhs: FractionF64 = rhs.into();
                self.sub(&rhs)
            }
        }
    };
}

macro_rules! sub_assign {
    ($t:ident) => {
        impl SubAssign<$t> for FractionF64 {
            fn sub_assign(&mut self, rhs: $t) {
                let rhs: FractionF64 = rhs.into();
                self.sub_assign(rhs)
            }
        }
    };
}

macro_rules! mul {
    ($t:ident) => {
        impl<'a> Mul<$t> for &'a FractionF64 {
            type Output = FractionF64;

            fn mul(self, rhs: $t) -> Self::Output {
                let rhs: FractionF64 = rhs.into();
                self.mul(&rhs)
            }
        }
    };
}

macro_rules! mul_assign {
    ($t:ident) => {
        impl MulAssign<$t> for FractionF64 {
            fn mul_assign(&mut self, rhs: $t) {
                let rhs: FractionF64 = rhs.into();
                self.mul_assign(rhs)
            }
        }
    };
}

macro_rules! div {
    ($t:ident) => {
        impl<'a> Div<$t> for &'a FractionF64 {
            type Output = FractionF64;

            fn div(self, rhs: $t) -> Self::Output {
                let rhs: FractionF64 = rhs.into();
                self.div(&rhs)
            }
        }
    };
}

macro_rules! div_assign {
    ($t:ident) => {
        impl DivAssign<$t> for FractionF64 {
            fn div_assign(&mut self, rhs: $t) {
                let rhs: FractionF64 = rhs.into();
                self.div_assign(rhs)
            }
        }
    };
}

macro_rules! ttype_tuple {
    ($t:ident) => {
        from_tuple_u_u!($t, usize);
        from_tuple_u_u!($t, u128);
        from_tuple_u_u!($t, u64);
        from_tuple_u_u!($t, u32);
        from_tuple_u_u!($t, u16);
        from_tuple_u_u!($t, u8);
        from_tuple_u_i!($t, i128);
        from_tuple_u_i!($t, i64);
        from_tuple_u_i!($t, i32);
        from_tuple_u_i!($t, i16);
        from_tuple_u_i!($t, i8);
    };
}

macro_rules! ttype_tuple_signed {
    ($t:ident) => {
        from_tuple_i_u!($t, usize);
        from_tuple_i_u!($t, u128);
        from_tuple_i_u!($t, u64);
        from_tuple_i_u!($t, u32);
        from_tuple_i_u!($t, u16);
        from_tuple_i_u!($t, u8);
        from_tuple_i_i!($t, i64);
        from_tuple_i_i!($t, i32);
        from_tuple_i_i!($t, i16);
        from_tuple_i_i!($t, i8);
    };
}

macro_rules! ttype {
    ($t:ident) => {
        from!($t);
        ttype_tuple!($t);
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
        from_signed!($t);
        ttype_tuple_signed!($t);
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
    use std::ops::Neg;

    use crate::{
        ebi_number::{One, Signed},
        fraction::fraction_f64::FractionF64,
    };

    #[test]
    fn fraction_neg() {
        let one = FractionF64::one();
        assert!(one.is_positive());
        let one = one.neg();
        assert!(one.is_negative());
    }

    #[test]
    fn fraction_parse() {
        let x = "0.2".to_owned();
        let f: FractionF64 = x.parse().unwrap();
        assert_eq!(f, FractionF64::from((1, 5)));

        assert_eq!("1".parse::<FractionF64>().unwrap(), FractionF64::one());
        assert_eq!("-1".parse::<FractionF64>().unwrap(), -FractionF64::one());

        assert_eq!("1.00".parse::<FractionF64>().unwrap(), FractionF64::one());
        assert_eq!("-1.00".parse::<FractionF64>().unwrap(), -FractionF64::one());

        assert_eq!(
            "1/5".parse::<FractionF64>().unwrap(),
            FractionF64::from((1, 5))
        );
        assert_eq!(
            "-1/5".parse::<FractionF64>().unwrap(),
            -FractionF64::from((1, 5))
        );

        assert_eq!(
            ".2".parse::<FractionF64>().unwrap(),
            FractionF64::from((1, 5))
        );
        assert_eq!(
            "-.2".parse::<FractionF64>().unwrap(),
            -FractionF64::from((1, 5))
        );
    }
}
