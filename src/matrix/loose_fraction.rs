use std::ops::{Add, Mul, Neg};

use fraction::{GenericFraction, Sign};
use num::{BigUint, Integer};

///A loose fraction is a sign, numerator and denominator. It is not necessary reduced.
pub trait LooseFraction<T, U> {
    /// Given three numbers a, b, and c, performs:
    /// a += b * c
    fn add_assign_mul(
        type_a: &mut Type,
        num_a: &mut Self,
        den_a: &mut Self,
        type_b: Type,
        num_b: &T,
        den_b: &T,
        type_c: Type,
        num_c: &U,
        den_c: &U,
    );

    // fn mul_assign<'a>(a: FractionRawMut<'a, T>, b: &FractionRaw<T>);
}

macro_rules! checked_mul {
    ($a: expr, $b: expr) => {
        if let Some(v) = $a.checked_mul($b) {
            v
        } else {
            return false;
        }
    };
}

macro_rules! checked_add {
    ($a: expr, $b: expr) => {
        if let Some(v) = $a.checked_add($b) {
            v
        } else {
            return false;
        }
    };
}

/// Given three numbers a, b, and c, performs:
/// a += b * c
///
/// Returns false if there was an overflow. In that case, the first three arguments are left in a non-determined state.
pub fn checked_add_assign_mul(
    type_a: &mut Type,
    num_a: &mut u64,
    den_a: &mut u64,
    type_b: Type,
    num_b: &u64,
    den_b: &u64,
    type_c: Type,
    num_c: &u64,
    den_c: &u64,
) -> bool {
    let type_prod = type_b * type_c;
    if let Some(new_type) = *type_a + type_prod {
        //the result type has been decided
        *type_a = new_type;

        if type_a.is_plusminus() {
            //type_a already contains the correct sign, so we can just add
            let num_prod = checked_mul!(num_b, *num_c);
            let den_prod = checked_mul!(den_b, *den_c);

            //numerator
            *num_a = checked_mul!(num_a, den_prod);
            *num_a = checked_add!(num_a, checked_mul!(den_a, num_prod));

            //denominator
            *den_a = checked_mul!(den_a, den_prod);
        } else {
            //type_a is not a non-positive or non-negative number, so the type determines it fully already.
        }
    } else {
        //one of the numbers is non-negative; the other is non-positive

        //compute the product
        let num_prod = checked_mul!(num_b, *num_c);
        let den_prod = checked_mul!(den_b, *den_c);

        match (&*type_a, type_prod) {
            (Type::Plus, Type::Minus) => {
                //numerator
                *num_a = checked_mul!(num_a, den_prod);
                let mut num_prod_adjusted = checked_mul!(&*den_a, num_prod);

                if &*num_a >= &num_prod_adjusted {
                    *type_a = Type::Plus;
                    *num_a -= num_prod_adjusted;
                } else {
                    *type_a = Type::Minus;
                    std::mem::swap(num_a, &mut num_prod_adjusted);
                    *num_a -= num_prod_adjusted;
                }

                //denominator
                *den_a = checked_mul!(den_a, den_prod);
            }
            (Type::Minus, Type::Plus) => {
                //numerator
                *num_a = checked_mul!(num_a, den_prod);
                let mut num_prod_adjusted = checked_mul!(den_a, num_prod);

                if &*num_a >= &num_prod_adjusted {
                    *type_a = Type::Minus;
                    *num_a -= num_prod_adjusted;
                } else {
                    *type_a = Type::Minus;
                    std::mem::swap(num_a, &mut num_prod_adjusted);
                    *num_a -= num_prod_adjusted;
                }

                //denominator
                *den_a *= den_prod;
            }
            _ => unreachable!(),
        }
    }
    true
}

macro_rules! aam {
    ($t:ident, $u:ident) => {
        impl LooseFraction<$t, $u> for BigUint {
            fn add_assign_mul(
                type_a: &mut Type,
                num_a: &mut Self,
                den_a: &mut Self,
                type_b: Type,
                num_b: &$t,
                den_b: &$t,
                type_c: Type,
                num_c: &$u,
                den_c: &$u,
            ) {
                let type_prod = type_b * type_c;
                if let Some(new_type) = *type_a + type_prod {
                    //the result type has been decided
                    *type_a = new_type;

                    if type_a.is_plusminus() {
                        //type_a already contains the correct sign, so we can just add
                        let num_prod = num_b * num_c;
                        let den_prod = den_b * den_c;

                        //numerator
                        *num_a *= &den_prod;
                        *num_a += &*den_a * &num_prod;

                        //denominator
                        *den_a *= den_prod;
                    } else {
                        //type_a is not a non-positive or non-negative number, so the type determines it fully already.
                    }
                } else {
                    //one of the numbers is non-negative; the other is non-positive

                    //compute the product
                    let num_prod = num_b * num_c;
                    let den_prod = den_b * den_c;

                    //do the addition
                    match (&*type_a, type_prod) {
                        (Type::Plus, Type::Minus) => {
                            //numerator
                            *num_a *= &den_prod;
                            let mut num_prod_adjusted = &*den_a * &num_prod;

                            if &*num_a >= &num_prod_adjusted {
                                *type_a = Type::Plus;
                                *num_a -= num_prod_adjusted;
                            } else {
                                *type_a = Type::Minus;
                                std::mem::swap(num_a, &mut num_prod_adjusted);
                                *num_a -= num_prod_adjusted;
                            }

                            //denominator
                            *den_a *= den_prod;
                        }
                        (Type::Minus, Type::Plus) => {
                            //numerator
                            *num_a *= &den_prod;
                            let mut num_prod_adjusted = &*den_a * &num_prod;

                            if &*num_a >= &num_prod_adjusted {
                                *type_a = Type::Minus;
                                *num_a -= num_prod_adjusted;
                            } else {
                                *type_a = Type::Minus;
                                std::mem::swap(num_a, &mut num_prod_adjusted);
                                *num_a -= num_prod_adjusted;
                            }

                            //denominator
                            *den_a *= den_prod;
                        }
                        _ => unreachable!(),
                    }
                }
            }
        }
    };
}

aam!(BigUint, BigUint);
aam!(BigUint, u64);
aam!(u64, BigUint);

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Type {
    Plus,
    Minus,
    NaN,
    Infinite,
    NegInfinite,
}

impl Type {
    pub fn sign(&self) -> Sign {
        match self {
            Type::Plus => Sign::Plus,
            Type::Minus => Sign::Minus,
            Type::NaN => Sign::Plus,
            Type::Infinite => Sign::Plus,
            Type::NegInfinite => Sign::Minus,
        }
    }

    pub fn is_plusminus(&self) -> bool {
        match self {
            Type::Plus => true,
            Type::Minus => true,
            Type::NaN => false,
            Type::Infinite => false,
            Type::NegInfinite => false,
        }
    }
}

impl<T: Clone + Integer> From<&GenericFraction<T>> for Type {
    fn from(value: &GenericFraction<T>) -> Self {
        match value {
            fraction::GenericFraction::Rational(Sign::Plus, _) => Type::Plus,
            fraction::GenericFraction::Rational(Sign::Minus, _) => Type::Minus,
            fraction::GenericFraction::Infinity(Sign::Plus) => Type::Infinite,
            fraction::GenericFraction::Infinity(Sign::Minus) => Type::NegInfinite,
            fraction::GenericFraction::NaN => Type::NaN,
        }
    }
}

impl Mul for Type {
    type Output = Type;

    fn mul(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Type::Plus, Type::Plus) => Type::Plus,
            (Type::Plus, Type::Minus) => Type::Minus,
            (Type::Plus, Type::Infinite) => Type::Infinite,
            (Type::Plus, Type::NegInfinite) => Type::NegInfinite,
            (Type::Minus, Type::Plus) => Type::Minus,
            (Type::Minus, Type::Minus) => Type::Plus,
            (Type::Minus, Type::Infinite) => Type::NegInfinite,
            (Type::Minus, Type::NegInfinite) => Type::Infinite,
            (_, Type::NaN) => Type::NaN,
            (Type::NaN, _) => Type::NaN,
            (Type::Infinite, Type::Plus) => Type::Infinite,
            (Type::Infinite, Type::Minus) => Type::NegInfinite,
            (Type::Infinite, Type::Infinite) => Type::Infinite,
            (Type::Infinite, Type::NegInfinite) => Type::NegInfinite,
            (Type::NegInfinite, Type::Plus) => Type::NegInfinite,
            (Type::NegInfinite, Type::Minus) => Type::Infinite,
            (Type::NegInfinite, Type::Infinite) => Type::NegInfinite,
            (Type::NegInfinite, Type::NegInfinite) => Type::Infinite,
        }
    }
}

impl Add for Type {
    type Output = Option<Type>;

    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Type::NaN, _) => Some(Type::NaN),
            (_, Type::NaN) => Some(Type::NaN),
            (Type::Plus, Type::Plus) => Some(Type::Plus),
            (Type::Plus, Type::Minus) => None,
            (Type::Plus, Type::Infinite) => Some(Type::Infinite),
            (Type::Plus, Type::NegInfinite) => Some(Type::NegInfinite),
            (Type::Minus, Type::Plus) => None,
            (Type::Minus, Type::Minus) => Some(Type::Minus),
            (Type::Minus, Type::Infinite) => Some(Type::Infinite),
            (Type::Minus, Type::NegInfinite) => Some(Type::NegInfinite),
            (Type::Infinite, Type::Plus) => Some(Type::Infinite),
            (Type::Infinite, Type::Minus) => Some(Type::Infinite),
            (Type::Infinite, Type::Infinite) => Some(Type::Infinite),
            (Type::Infinite, Type::NegInfinite) => Some(Type::NaN),
            (Type::NegInfinite, Type::Plus) => Some(Type::NegInfinite),
            (Type::NegInfinite, Type::Minus) => Some(Type::NegInfinite),
            (Type::NegInfinite, Type::Infinite) => Some(Type::NaN),
            (Type::NegInfinite, Type::NegInfinite) => Some(Type::NegInfinite),
        }
    }
}

impl Neg for Type {
    type Output = Type;

    fn neg(self) -> Self::Output {
        match self {
            Type::Plus => Type::Minus,
            Type::Minus => Type::Plus,
            Type::NaN => Type::NaN,
            Type::Infinite => Type::NegInfinite,
            Type::NegInfinite => Type::Infinite,
        }
    }
}
