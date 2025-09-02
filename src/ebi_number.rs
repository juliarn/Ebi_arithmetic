use crate::fraction::{
    fraction_enum::FractionEnum, fraction_exact::FractionExact, fraction_f64::FractionF64,
};
use anyhow::Result;

pub trait EbiNumber: Zero + One + Round + Clone {}

impl EbiNumber for FractionEnum {}
impl EbiNumber for FractionF64 {}
impl EbiNumber for FractionExact {}
impl EbiNumber for f32 {}
impl EbiNumber for f64 {}
impl EbiNumber for usize {}
impl EbiNumber for u128 {}
impl EbiNumber for u64 {}
impl EbiNumber for u32 {}
impl EbiNumber for u16 {}
impl EbiNumber for u8 {}
impl EbiNumber for i128 {}
impl EbiNumber for i64 {}
impl EbiNumber for i32 {}
impl EbiNumber for i16 {}
impl EbiNumber for i8 {}

pub trait Zero: Sized {
    fn zero() -> Self;

    fn set_zero(&mut self) {
        *self = Zero::zero();
    }

    fn is_zero(&self) -> bool;
}

pub trait One: Sized {
    fn one() -> Self;

    fn set_one(&mut self) {
        *self = One::one();
    }

    fn is_one(&self) -> bool;
}

pub trait Signed: Sized {
    fn abs(self) -> Self;

    /// Returns true if the number is positive and false if the number is zero or negative.
    fn is_positive(&self) -> bool;

    /// Returns true if the number is negative and false if the number is zero or positive.
    fn is_negative(&self) -> bool;

    /// For exact arithmetic: Returns true if the number is positive or zero.
    /// For approximate arithmetic: returns true if the number is larger than -epsilon
    fn is_not_negative(&self) -> bool {
        !self.is_negative()
    }

    /// For exact arithmetic: Returns true if the number is negative or zero.
    /// For approximate arithmetic: returns true if the number is smaller than epsilon
    fn is_not_positive(&self) -> bool {
        !self.is_positive()
    }
}

pub trait Round: Sized {
    /// Returns the largest integer less than or equal to `self`.
    fn floor(self) -> Self;

    /// Returns the smallest integer greater than or equal to `self`.
    fn ceil(self) -> Self;
}

pub trait Recip: Sized {
    /// Takes the reciprocal (inverse) of a number, `1/x`.
    fn recip(self) -> Self;
}

pub trait OneMinus: Sized {
    fn one_minus(self) -> Self;
}

pub trait ChooseRandomly {
    type Cache;

    /// Return a random index from 0 (inclusive) to the length of the list (exclusive).
    /// The likelihood of each index to be returned is proportional to the value of the fraction at that index.
    ///
    /// The fractions do not need to sum to 1, and do not need to be sorted, but need to be positive.
    ///
    /// If more than a couple of draws are made, consider creating a cache and drawing from it.
    fn choose_randomly(fractions: &Vec<Self>) -> Result<usize>
    where
        Self: Sized;

    fn choose_randomly_create_cache<'a>(
        fractions: impl Iterator<Item = &'a Self>,
    ) -> Result<Self::Cache>
    where
        Self: Sized,
        Self: 'a;

    fn choose_randomly_cached(cache: &Self::Cache) -> usize
    where
        Self: Sized;
}

pub trait Sqrt {
    /// # Calculates the approximate square root of the value
    ///
    /// Calculates the approximate square root of `value`.  If the returned value is
    /// `Ok(_)`, then it is guaranteed to be within `epsilon` of the actual
    /// answer.  If `epsilon <= 0.0`, then `Err` is returned (the reason for the
    /// bound of `0.0` is because the approximation algorithm is unable to return an
    /// exact answer).  If `value < 0.0`, then `Err` is returned (`BigRational` is
    /// a real valued object; it cannot represent complex values).  In both `Err`
    /// cases, the value will be a `String` explaining what the error actually is.
    ///
    /// # Parameters
    ///
    /// - `value` - The value whose approximate square root you wish to obtain.  If
    ///     this is less than `0.0`, then `Err` will be returned.
    /// - `epsilon` - The maximum acceptable difference between the returned value
    ///     and the actual value.  The returned value is in the range
    ///     `[actual - 1/10^decimals, actual + 1/10^decimals]`.
    ///
    /// # Returns
    ///
    /// If everything went as expected, then `Ok(_)` will be returned, containing
    /// a value that is within `± epsilon` of the actual value.  If anything went
    /// wrong, then `Err(_)` will be returned, containing a `String` outlining what
    /// the problem was.
    fn approx_sqrt(&self, precision_decimals: u32) -> Result<Self>
    where
        Self: Sized;

    fn approx_abs_sqrt(self, precision_decimals: u32) -> Self
    where
        Self: Sized + Signed,
    {
        self.abs().approx_sqrt(precision_decimals).unwrap()
    }
}
