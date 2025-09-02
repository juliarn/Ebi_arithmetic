use anyhow::{Error, anyhow};
use malachite::{Integer, Natural, rational::Rational};
use std::{
    borrow::Borrow,
    cmp::Ordering,
    hash::Hash,
    iter::Sum,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    str::FromStr,
    sync::Arc,
};

use crate::ebi_number::Zero;

#[derive(Clone)]
pub struct FractionExact(pub(crate) Rational);

impl Default for FractionExact {
    fn default() -> Self {
        Self::zero()
    }
}

//======================== from ========================//

impl FromStr for FractionExact {
    type Err = Error;

    fn from_str(s: &str) -> std::prelude::v1::Result<Self, Self::Err> {
        //integer
        if let Ok(n) = s.parse::<Integer>() {
            return Ok(Self(n.into()));
        }

        //float
        {
            if s.starts_with('-') {
                //negative
                let s = &s[1..];
                let mut arr = s.split('.');
                if let Some(mut int) = arr.next() {
                    if int.len() == 0 {
                        int = &"0";
                    }
                    if let Some(dec) = arr.next() {
                        let digits = dec.len() as u32;
                        match (int.parse::<Natural>(), dec.parse::<Natural>()) {
                            (Ok(num), Ok(den)) => {
                                return Ok(-Self(
                                    Rational::from(num)
                                        + Rational::from(den) / Rational::from(10_u64.pow(digits)),
                                ));
                            }
                            _ => {}
                        }
                    }
                }
            } else {
                //positive
                let mut arr = s.split('.');
                if let Some(mut int) = arr.next() {
                    if int.len() == 0 {
                        int = &"0";
                    }
                    if let Some(dec) = arr.next() {
                        let digits = dec.len() as u32;
                        match (int.parse::<Natural>(), dec.parse::<Natural>()) {
                            (Ok(num), Ok(den)) => {
                                return Ok(Self(
                                    Rational::from(num)
                                        + Rational::from(den) / Rational::from(10_u64.pow(digits)),
                                ));
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        //fraction
        Ok(Self(match Rational::from_str(s) {
            Ok(x) => x,
            Err(_) => {
                return Err(anyhow!("parsing fraction failed"));
            }
        }))
    }
}

impl From<&FractionExact> for FractionExact {
    fn from(value: &FractionExact) -> Self {
        value.clone()
    }
}

impl From<Arc<FractionExact>> for FractionExact {
    fn from(value: Arc<FractionExact>) -> Self {
        Self(value.0.clone())
    }
}

impl From<&Arc<FractionExact>> for FractionExact {
    fn from(value: &Arc<FractionExact>) -> Self {
        match value.as_ref() {
            FractionExact(f) => FractionExact(f.clone()),
        }
    }
}

impl From<Rational> for FractionExact {
    fn from(value: Rational) -> Self {
        Self(value)
    }
}

impl From<Natural> for FractionExact {
    fn from(value: Natural) -> Self {
        Self(value.into())
    }
}

impl From<Integer> for FractionExact {
    fn from(value: Integer) -> Self {
        Self(value.into())
    }
}

macro_rules! from_1 {
    ($t:ident, $u:ident) => {
        impl From<($t, $u)> for FractionExact {
            fn from(value: ($t, $u)) -> Self {
                Self(Rational::from(value.0) / Rational::from(value.1))
            }
        }
    };
}

macro_rules! from_2 {
    ($t:ident) => {
        impl From<$t> for FractionExact {
            fn from(value: $t) -> Self {
                Self(Rational::from(value))
            }
        }

        // from_1!($t, Integer);
        from_1!($t, usize);
        from_1!($t, u8);
        from_1!($t, u16);
        from_1!($t, u32);
        from_1!($t, u64);
        from_1!($t, u128);
        from_1!($t, i8);
        from_1!($t, i16);
        from_1!($t, i32);
        from_1!($t, i64);
        from_1!($t, i128);
    };
}

macro_rules! from_primitive {
    ($t:ident) => {
        from_2!($t);

        impl From<&$t> for FractionExact {
            fn from(value: &$t) -> Self {
                Self(Rational::from(*value))
            }
        }
    };
}

from_primitive!(usize);
from_primitive!(u8);
from_primitive!(u16);
from_primitive!(u32);
from_primitive!(u64);
from_primitive!(u128);
from_primitive!(i8);
from_primitive!(i16);
from_primitive!(i32);
from_primitive!(i64);
from_primitive!(i128);

//======================== shorthand macros ========================//

#[macro_export]
/// Convenience short-hand macro to create fractions.
macro_rules! f_e {
    ($e: expr) => {
        FractionExact::from($e)
    };

    ($e: expr, $f: expr) => {
        FractionExact::from(($e, $f))
    };
}
pub use f_e;

#[macro_export]
/// Convenience short-hand macro to create a fraction representing zero.
macro_rules! f0_e {
    () => {
        FractionExact::zero()
    };
}
pub use f0_e;

#[macro_export]
/// Convenience short-hand macro to create a fraction representing one.
macro_rules! f1_e {
    () => {
        FractionExact::one()
    };
}
pub use f1_e;

impl std::fmt::Display for FractionExact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}

impl std::fmt::Debug for FractionExact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Exact ").field(&self.0).finish()
    }
}

//======================== operators ========================//

impl Add for FractionExact {
    type Output = FractionExact;

    fn add(self, rhs: Self) -> Self::Output {
        FractionExact(self.0 + rhs.0)
    }
}

impl Add<&FractionExact> for &FractionExact {
    type Output = FractionExact;

    fn add(self, rhs: &FractionExact) -> Self::Output {
        FractionExact(&self.0 + &rhs.0)
    }
}

impl Sub for FractionExact {
    type Output = FractionExact;

    fn sub(self, rhs: Self) -> Self::Output {
        FractionExact(self.0 - rhs.0)
    }
}

impl Sub<&FractionExact> for &FractionExact {
    type Output = FractionExact;

    fn sub(self, rhs: &FractionExact) -> Self::Output {
        FractionExact(&self.0 - &rhs.0)
    }
}

impl Div for FractionExact {
    type Output = FractionExact;

    fn div(self, rhs: Self) -> Self::Output {
        FractionExact(self.0 / rhs.0)
    }
}

impl Div<&FractionExact> for &FractionExact {
    type Output = FractionExact;

    fn div(self, rhs: &FractionExact) -> Self::Output {
        FractionExact(&self.0 / &rhs.0)
    }
}

impl Mul for FractionExact {
    type Output = FractionExact;

    fn mul(self, rhs: Self) -> Self::Output {
        FractionExact(self.0 * rhs.0)
    }
}

impl Mul<&FractionExact> for &FractionExact {
    type Output = FractionExact;

    fn mul(self, rhs: &FractionExact) -> Self::Output {
        FractionExact(&self.0 * &rhs.0)
    }
}

macro_rules! binary_operator {
    ($t:ident) => {
        impl Add<$t> for FractionExact {
            type Output = FractionExact;

            fn add(self, rhs: $t) -> Self::Output {
                Self(self.0 + Rational::from(rhs))
            }
        }

        impl Add<&$t> for FractionExact {
            type Output = FractionExact;

            fn add(self, rhs: &$t) -> Self::Output {
                Self(self.0 + Rational::from(*rhs))
            }
        }

        impl Sub<$t> for FractionExact {
            type Output = FractionExact;

            fn sub(self, rhs: $t) -> Self::Output {
                Self(self.0 - Rational::from(rhs))
            }
        }

        impl Sub<&$t> for FractionExact {
            type Output = FractionExact;

            fn sub(self, rhs: &$t) -> Self::Output {
                Self(self.0 - Rational::from(*rhs))
            }
        }

        impl Div<$t> for FractionExact {
            type Output = FractionExact;

            fn div(self, rhs: $t) -> Self::Output {
                Self(self.0 / Rational::from(rhs))
            }
        }

        impl Div<&$t> for FractionExact {
            type Output = FractionExact;

            fn div(self, rhs: &$t) -> Self::Output {
                Self(self.0 / Rational::from(*rhs))
            }
        }

        impl Mul<$t> for FractionExact {
            type Output = FractionExact;

            fn mul(self, rhs: $t) -> Self::Output {
                Self(self.0 * Rational::from(rhs))
            }
        }

        impl Mul<&$t> for FractionExact {
            type Output = FractionExact;

            fn mul(self, rhs: &$t) -> Self::Output {
                Self(self.0 * Rational::from(*rhs))
            }
        }
    };
}

binary_operator!(usize);
binary_operator!(u8);
binary_operator!(u16);
binary_operator!(u32);
binary_operator!(u64);
binary_operator!(u128);
binary_operator!(i8);
binary_operator!(i16);
binary_operator!(i32);
binary_operator!(i64);
binary_operator!(i128);

//======================== assign operators ========================//

impl<T> AddAssign<T> for FractionExact
where
    T: Borrow<FractionExact>,
{
    fn add_assign(&mut self, rhs: T) {
        let rhs = rhs.borrow();
        match (self, rhs) {
            (FractionExact(x), FractionExact(y)) => x.add_assign(y),
        }
    }
}

impl AddAssign<&Arc<FractionExact>> for FractionExact {
    fn add_assign(&mut self, rhs: &Arc<FractionExact>) {
        let rhs = rhs.borrow();
        match (self, rhs) {
            (FractionExact(x), FractionExact(y)) => x.add_assign(y),
        }
    }
}

impl<T> SubAssign<T> for FractionExact
where
    T: Borrow<FractionExact>,
{
    fn sub_assign(&mut self, rhs: T) {
        let rhs = rhs.borrow();
        match (self, rhs) {
            (FractionExact(x), FractionExact(y)) => x.sub_assign(y),
        }
    }
}

impl<T> MulAssign<T> for FractionExact
where
    T: Borrow<FractionExact>,
{
    fn mul_assign(&mut self, rhs: T) {
        let rhs = rhs.borrow();
        match (self, rhs) {
            (FractionExact(x), FractionExact(y)) => x.mul_assign(y),
        }
    }
}

impl<T> DivAssign<T> for FractionExact
where
    T: Borrow<FractionExact>,
{
    fn div_assign(&mut self, rhs: T) {
        let rhs = rhs.borrow();
        match (self, rhs) {
            (FractionExact(x), FractionExact(y)) => x.div_assign(y),
        }
    }
}

impl Neg for FractionExact {
    type Output = FractionExact;

    fn neg(self) -> Self::Output {
        FractionExact(self.0.neg())
    }
}

impl<'a> Neg for &'a FractionExact {
    type Output = FractionExact;

    fn neg(self) -> Self::Output {
        match self {
            FractionExact(f) => FractionExact(f.neg()),
        }
    }
}

impl PartialEq for FractionExact {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (FractionExact(x), FractionExact(y)) => x == y,
        }
    }
}

impl Eq for FractionExact {}

impl PartialOrd for FractionExact {
    /**
     * Note that exact and approximate should not be compared.
     */
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (FractionExact(x), FractionExact(y)) => x.partial_cmp(y),
        }
    }
}

impl Ord for FractionExact {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)
    }
}

impl Hash for FractionExact {
    /**
     * For good reasons, Rust does not support hashing of doubles. However, we need it to store distributions in a hashmap.
     * Approximate arithmetic is discouraged
     */
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            FractionExact(f) => f.hash(state),
        }
    }
}

impl Sum for FractionExact {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(<FractionExact as Zero>::zero(), |sum, f| sum + f)
    }
}

impl<'a> Sum<&'a FractionExact> for FractionExact {
    fn sum<I: Iterator<Item = &'a FractionExact>>(iter: I) -> Self {
        iter.fold(<FractionExact as Zero>::zero(), |sum, f| f + &sum)
    }
}

macro_rules! add {
    ($t:ident) => {
        impl<'a> Add<$t> for &'a FractionExact {
            type Output = FractionExact;

            fn add(self, rhs: $t) -> Self::Output {
                let rhs = rhs.into();
                match (self, rhs) {
                    (FractionExact(x), FractionExact(y)) => FractionExact(x.add(y)),
                }
            }
        }
    };
}

macro_rules! add_assign {
    ($t:ident) => {
        impl AddAssign<$t> for FractionExact {
            fn add_assign(&mut self, rhs: $t) {
                let rhs = rhs.into();
                match (self, rhs) {
                    (FractionExact(x), FractionExact(y)) => x.add_assign(y),
                }
            }
        }
    };
}

macro_rules! sub {
    ($t:ident) => {
        impl<'a> Sub<$t> for &'a FractionExact {
            type Output = FractionExact;

            fn sub(self, rhs: $t) -> Self::Output {
                let rhs = rhs.into();
                match (self, rhs) {
                    (FractionExact(x), FractionExact(y)) => FractionExact(x.sub(y)),
                }
            }
        }
    };
}

macro_rules! sub_assign {
    ($t:ident) => {
        impl SubAssign<$t> for FractionExact {
            fn sub_assign(&mut self, rhs: $t) {
                let rhs = rhs.into();
                match (self, rhs) {
                    (FractionExact(x), FractionExact(y)) => x.sub_assign(y),
                }
            }
        }
    };
}

macro_rules! mul {
    ($t:ident) => {
        impl<'a> Mul<$t> for &'a FractionExact {
            type Output = FractionExact;

            fn mul(self, rhs: $t) -> Self::Output {
                let rhs = rhs.into();
                match (self, rhs) {
                    (FractionExact(x), FractionExact(y)) => FractionExact(x.mul(y)),
                }
            }
        }
    };
}

macro_rules! mul_assign {
    ($t:ident) => {
        impl MulAssign<$t> for FractionExact {
            fn mul_assign(&mut self, rhs: $t) {
                let rhs = rhs.into();
                match (self, rhs) {
                    (FractionExact(x), FractionExact(y)) => x.mul_assign(y),
                }
            }
        }
    };
}

macro_rules! div {
    ($t:ident) => {
        impl<'a> Div<$t> for &'a FractionExact {
            type Output = FractionExact;

            fn div(self, rhs: $t) -> Self::Output {
                let rhs = rhs.into();
                match (self, rhs) {
                    (FractionExact(x), FractionExact(y)) => FractionExact(x.div(y)),
                }
            }
        }
    };
}

macro_rules! div_assign {
    ($t:ident) => {
        impl DivAssign<$t> for FractionExact {
            fn div_assign(&mut self, rhs: $t) {
                let rhs = rhs.into();
                match (self, rhs) {
                    (FractionExact(x), FractionExact(y)) => x.div_assign(y),
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
    use std::ops::Neg;

    use crate::{
        ebi_number::{One, Signed},
        fraction::fraction_exact::FractionExact,
    };

    #[test]
    fn fraction_neg() {
        let one = FractionExact::one();
        assert!(one.is_positive());
        let one = one.neg();
        assert!(one.is_negative());
    }

    #[test]
    fn fraction_parse() {
        let x = "0.2".to_owned();
        let f: FractionExact = x.parse().unwrap();
        assert_eq!(f, FractionExact::from((1, 5)));

        assert_eq!("1".parse::<FractionExact>().unwrap(), FractionExact::one());
        assert_eq!(
            "-1".parse::<FractionExact>().unwrap(),
            -FractionExact::one()
        );

        assert_eq!(
            "1.00".parse::<FractionExact>().unwrap(),
            FractionExact::one()
        );
        assert_eq!(
            "-1.00".parse::<FractionExact>().unwrap(),
            -FractionExact::one()
        );

        assert_eq!(
            "1/5".parse::<FractionExact>().unwrap(),
            FractionExact::from((1, 5))
        );
        assert_eq!(
            "-1/5".parse::<FractionExact>().unwrap(),
            -FractionExact::from((1, 5))
        );

        assert_eq!(
            ".2".parse::<FractionExact>().unwrap(),
            FractionExact::from((1, 5))
        );
        assert_eq!(
            "-.2".parse::<FractionExact>().unwrap(),
            -FractionExact::from((1, 5))
        );
    }
}
