use crate::{
    Recip, Signed, Sqrt,
    fraction::{
        fraction_enum::FractionEnum, fraction_exact::FractionExact, fraction_f64::FractionF64,
    },
};
use anyhow::{Result, anyhow};
use malachite::{
    Integer, Natural,
    base::num::{
        basic::traits::{One, Two, Zero},
        conversion::traits::IsInteger,
        logic::traits::SignificantBits,
    },
    rational::Rational,
};

impl Sqrt for FractionF64 {
    fn approx_sqrt(&self, precision_decimals: u32) -> Result<Self>
    where
        Self: Sized,
    {
        Ok(Self(self.0.approx_sqrt(precision_decimals)?))
    }
}

impl Sqrt for FractionExact {
    fn approx_sqrt(&self, precision_decimals: u32) -> Result<Self> {
        Ok(Self(self.0.approx_sqrt(precision_decimals)?))
    }
}

impl Sqrt for FractionEnum {
    fn approx_sqrt(&self, precision_decimals: u32) -> Result<Self>
    where
        Self: Sized,
    {
        match self {
            FractionEnum::Exact(f) => Ok(FractionEnum::Exact(f.approx_sqrt(precision_decimals)?)),
            FractionEnum::Approx(f) => Ok(FractionEnum::Approx(f.approx_sqrt(precision_decimals)?)),
            FractionEnum::CannotCombineExactAndApprox => {
                Err(anyhow!("cannot combine exact and approximate arithmetic"))
            }
        }
    }
}

impl Sqrt for f64 {
    fn approx_sqrt(&self, _precision_decimals: u32) -> Result<Self>
    where
        Self: Sized,
    {
        if *self < f64::ZERO {
            return Err(anyhow!(
                "cannot calculate the square root of negative values"
            ));
        }

        Ok(self.sqrt())
    }
}

impl Sqrt for Rational {
    /// With credits to Cem Karan, https://github.com/rust-num/num-rational/issues/35
    fn approx_sqrt(&self, precision_decimals: u32) -> Result<Self>
    where
        Self: Sized,
    {
        if *self < Rational::ZERO {
            return Err(anyhow!(
                "cannot calculate the square root of negative values"
            ));
        }

        // First try whether the result is an integer
        // Step 1: the number itself needs to be an integer
        if self.is_integer() {
            //perform binary search on the root
            let floor: Natural =
                malachite::base::num::arithmetic::traits::Ceiling::ceiling(self.clone())
                    .try_into()
                    .unwrap();
            let sqrt = sqrt_search(&Natural::ONE, &floor, &floor);
            if &sqrt * &sqrt == floor {
                return Ok(sqrt.into());
            }
        }

        let epsilon = Rational::ONE / Rational::from(10_u64.pow(precision_decimals));

        if epsilon <= Rational::ZERO {
            return Err(anyhow!(
                "cannot calculate the square root with a non-positive epsilon."
            ));
        }

        // I'm going to use the Babylonian method to find the square root.  This is
        // described at
        // https://en.wikipedia.org/wiki/Methods_of_computing_square_roots#Babylonian_method
        // To do so, I need to have an initial seed value that is the approximate
        // square root.  This will estimate will be refined until it is within
        // epsilon of the real value.

        // Calculates seed values for all values >= 1.0.  This is used below when
        // calculating the seed value.
        #[inline]
        fn calc_seed(value: &Rational) -> Rational {
            let bits = malachite::base::num::arithmetic::traits::Ceiling::ceiling(value)
                .significant_bits();
            let half_bits = bits / 2;
            let approximate = Integer::ONE << half_bits;
            Rational::from(approximate)
        }

        let mut x = if *self >= Rational::ONE {
            calc_seed(&self)
        } else {
            // Because the value is less than one, I can't use the trick above
            // directly.  Instead, I'm going to find the reciprocal, and then do the
            // trick above, and then use the reciprocal of that as the seed.
            calc_seed(&(self.clone().recip())).recip()
        };

        // We now have an initial seed.  Time to refine it until it is within
        // epsilon of the real value.  Inlined functions could probably be really
        // inlined, but this is slightly easier for me to read.

        #[inline]
        fn calc_next_x(value: &Rational, x: Rational) -> Rational {
            let two = Rational::TWO;
            (&x + (value / &x)) / two
        }

        #[inline]
        fn calc_approx_error(value: &Rational, x: &Rational) -> Rational {
            let two = Rational::TWO;
            ((value - (x * x)) / (x * two)).abs()
        }

        while calc_approx_error(&self, &x) > epsilon {
            x = calc_next_x(&self, x);
        }

        Ok(x)
    }
}

fn sqrt_search(low: &Natural, high: &Natural, n: &Natural) -> Natural {
    if low <= high {
        let mid = (low + high) / Natural::TWO;

        if (&(&mid * &mid) <= n) && ((&mid + Natural::ONE) * (&mid + Natural::ONE) > *n) {
            return mid;
        } else if &(&mid * &mid) < n {
            return sqrt_search(&(mid + Natural::ONE), high, n);
        } else {
            return sqrt_search(low, &(mid - Natural::ONE), n);
        }
    }
    return low.clone();
}

#[cfg(test)]
mod test {
    use malachite::rational::Rational;

    use crate::Sqrt;

    #[test]
    fn sqrt_exact() {
        let three = Rational::from(3);
        let nine = Rational::from(9);
        let two = Rational::from(2);
        let sqrttwo = Rational::from(577) / Rational::from(408);

        assert_eq!(nine.approx_sqrt(4).unwrap(), three);

        assert_eq!(two.approx_sqrt(4).unwrap(), sqrttwo);
    }
}
