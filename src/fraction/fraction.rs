//======================== set type alias based on compile flags ========================//

#[cfg(any(
    all(
        not(feature = "exactarithmetic"),
        not(feature = "approximatearithmetic")
    ),
    all(feature = "exactarithmetic", feature = "approximatearithmetic")
))]
pub type Fraction = super::fraction_enum::FractionEnum;

#[cfg(all(not(feature = "exactarithmetic"), feature = "approximatearithmetic"))]
pub type Fraction = super::fraction_f64::FractionF64;

#[cfg(all(feature = "exactarithmetic", not(feature = "approximatearithmetic")))]
pub type Fraction = super::fraction_exact::FractionExact;

//======================== fraction tools ========================//

pub const APPROX_DIGITS: u64 = 5;
pub const EPSILON: f64 = 1e-13;

#[macro_export]
/// Convenience short-hand macro to create fractions.
macro_rules! f {
    ($e: expr) => {
        Fraction::from($e)
    };

    ($e: expr, $f: expr) => {
        Fraction::from(($e, $f))
    };
}
pub use f;

#[macro_export]
/// Convenience short-hand macro to create a fraction representing zero.
macro_rules! f0 {
    () => {
        Fraction::zero()
    };
}
pub use f0;

#[macro_export]
/// Convenience short-hand macro to create a fraction representing one.
macro_rules! f1 {
    () => {
        Fraction::one()
    };
}
pub use f1;
