use anyhow::{Context, Result, anyhow};
use malachite::{
    Natural, base::random::Seed, natural::random::random_naturals_less_than, rational::Rational,
};
use rand::{Rng, RngCore};

use crate::{
    ebi_number::{ChooseRandomly, Zero},
    exact::{MaybeExact, is_exact_globally},
    fraction::{
        fraction_enum::FractionEnum, fraction_exact::FractionExact, fraction_f64::FractionF64,
    },
};

#[cfg(any(
    all(
        not(feature = "exactarithmetic"),
        not(feature = "approximatearithmetic")
    ),
    all(feature = "exactarithmetic", feature = "approximatearithmetic")
))]
pub type FractionRandomCache = FractionRandomCacheEnum;

#[cfg(all(not(feature = "exactarithmetic"), feature = "approximatearithmetic"))]
pub type FractionRandomCache = FractionRandomCacheF64;

#[cfg(all(feature = "exactarithmetic", not(feature = "approximatearithmetic")))]
pub type FractionRandomCache = FractionRandomCacheExact;

pub enum FractionRandomCacheEnum {
    Exact(Vec<Rational>, Natural),
    Approx(Vec<f64>),
}

impl ChooseRandomly for FractionEnum {
    type Cache = FractionRandomCacheEnum;

    fn choose_randomly(fractions: &Vec<FractionEnum>) -> Result<usize> {
        if fractions.is_empty() {
            return Err(anyhow!("cannot take an element of an empty list"));
        }

        //normalise the inputs such that they sum to one.
        let mut probabilities: Vec<FractionEnum> = fractions.iter().cloned().collect();
        let sum = probabilities
            .iter()
            .fold(FractionEnum::zero(), |x, y| &x + y);
        if sum.is_zero() {
            return Err(anyhow!("sum of fractions is zero"));
        }
        if sum == FractionEnum::CannotCombineExactAndApprox {
            return Err(anyhow!("cannot combine exact and approximate arithmetic"));
        }
        probabilities.retain_mut(|v| {
            *v /= &sum;
            true
        });

        let mut rng = rand::rng();

        //select a random value
        let rand_val = if sum.is_exact() {
            let mut buf = [0u8; 32];
            rng.fill_bytes(&mut buf);
            let seed = Seed::from_bytes(buf);

            //strategy: the highest denominator determines how much precision we need
            let max_denom = probabilities
                .iter()
                .map(|f| match f {
                    FractionEnum::Exact(e) => e.to_denominator(),
                    _ => unreachable!(),
                })
                .max()
                .unwrap();
            //Generate a random value with the number of bits of the highest denominator. Repeat until this value is <= the max denominator.
            let rand_val = random_naturals_less_than(seed, max_denom.clone())
                .next()
                .unwrap();
            //create the fraction from the random nominator and the max denominator
            FractionEnum::Exact(Rational::from(rand_val) / Rational::from(max_denom.clone()))
        } else {
            //approximate mode
            FractionEnum::Approx(rng.random_range(0.0..=1.0))
        };

        let mut cum_prob = FractionEnum::zero();
        for (index, value) in probabilities.iter().enumerate() {
            cum_prob += value;
            if rand_val < cum_prob {
                return Ok(index);
            }
        }
        Ok(probabilities.len() - 1)
    }

    fn choose_randomly_create_cache<'a>(
        mut fractions: impl Iterator<Item = &'a Self>,
    ) -> Result<FractionRandomCacheEnum>
    where
        Self: Sized,
        Self: 'a,
    {
        if is_exact_globally() {
            //exact mode
            if let Some(first) = fractions.next() {
                let mut cumulative_probabilities = vec![
                    first
                        .exact_ref()
                        .with_context(|| "cannot combine exact and approximate arithmetic")?
                        .clone(),
                ];
                let mut highest_denom = first.exact_ref()?.to_denominator();

                while let Some(fraction) = fractions.next() {
                    highest_denom = highest_denom.max(fraction.exact_ref()?.to_denominator());

                    let mut x = fraction
                        .exact_ref()
                        .with_context(|| "cannot combine exact and approximate arithmetic")?
                        .clone();
                    x += cumulative_probabilities.last().unwrap();
                    cumulative_probabilities.push(x);
                }
                let highest_denom = highest_denom.clone();

                Ok(FractionRandomCacheEnum::Exact(
                    cumulative_probabilities,
                    highest_denom,
                ))
            } else {
                Err(anyhow!("cannot take an element of an empty list"))
            }
        } else {
            //approximate mode
            if let Some(first) = fractions.next() {
                let mut cumulative_probabilities = vec![
                    *first
                        .approx_ref()
                        .with_context(|| "cannot combine exact and approximate arithmetic")?,
                ];

                while let Some(fraction) = fractions.next() {
                    cumulative_probabilities.push(
                        fraction
                            .approx_ref()
                            .with_context(|| "cannot combine exact and approximate arithmetic")?
                            + cumulative_probabilities.last().unwrap(),
                    );
                }

                Ok(FractionRandomCacheEnum::Approx(cumulative_probabilities))
            } else {
                Err(anyhow!("cannot take an element of an empty list"))
            }
        }
    }

    fn choose_randomly_cached(cache: &FractionRandomCacheEnum) -> usize
    where
        Self: Sized,
    {
        match cache {
            FractionRandomCacheEnum::Exact(cumulative_probabilities, highest_denom) => {
                //select a random value
                let mut rng = rand::rng();
                let mut buf = [0u8; 32];
                rng.fill_bytes(&mut buf);
                let seed = Seed::from_bytes(buf);
                let rand_val = {
                    //strategy: the highest denominator determines how much precision we need

                    //Generate a random value with the number of bits of the highest denominator. Repeat until this value is <= the max denominator.
                    let rand_val = random_naturals_less_than(seed, highest_denom.clone())
                        .next()
                        .unwrap();

                    //create the fraction from the random nominator and the max denominator
                    Rational::from(rand_val) / Rational::from(highest_denom.clone())
                };

                match cumulative_probabilities.binary_search(&rand_val) {
                    Ok(index) | Err(index) => index,
                }
            }
            FractionRandomCacheEnum::Approx(cumulative_probabilities) => {
                //select a random value
                let mut rng = rand::rng();
                let rand_val = rng.random_range(0.0..=*cumulative_probabilities.last().unwrap());

                match cumulative_probabilities.binary_search_by(|probe| probe.total_cmp(&rand_val))
                {
                    Ok(index) | Err(index) => index,
                }
            }
        }
    }
}

pub struct FractionRandomCacheExact {
    cumulative_probabilities: Vec<FractionExact>,
    highest_denom: Natural,
}

impl ChooseRandomly for FractionExact {
    type Cache = FractionRandomCacheExact;

    fn choose_randomly(fractions: &Vec<FractionExact>) -> Result<usize> {
        if fractions.is_empty() {
            return Err(anyhow!("cannot take an element of an empty list"));
        }

        //normalise the inputs such that they sum to one.
        let mut probabilities: Vec<FractionExact> = fractions.iter().cloned().collect();
        let sum = probabilities
            .iter()
            .fold(FractionExact::zero(), |x, y| &x + y);
        if sum.is_zero() {
            return Err(anyhow!("sum of fractions is zero"));
        }
        probabilities.retain_mut(|v| {
            *v /= &sum;
            true
        });

        //select a random value
        let mut rng = rand::rng();
        let mut buf = [0u8; 32];
        rng.fill_bytes(&mut buf);
        let seed = Seed::from_bytes(buf);
        let rand_val = {
            //strategy: the highest denominator determines how much precision we need
            let max_denom = probabilities
                .iter()
                .map(|f| match f {
                    FractionExact(e) => e.to_denominator(),
                })
                .max()
                .unwrap();
            //Generate a random value with the number of bits of the highest denominator. Repeat until this value is <= the max denominator.
            let rand_val = random_naturals_less_than(seed, max_denom.clone())
                .next()
                .unwrap();
            //create the fraction from the random nominator and the max denominator
            FractionExact(Rational::from(rand_val) / Rational::from(max_denom.clone()))
        };

        let mut cum_prob = FractionExact::zero();
        for (index, value) in probabilities.iter().enumerate() {
            cum_prob += value;
            if rand_val < cum_prob {
                return Ok(index);
            }
        }
        Ok(probabilities.len() - 1)
    }

    fn choose_randomly_create_cache<'a>(
        mut fractions: impl Iterator<Item = &'a Self>,
    ) -> Result<FractionRandomCacheExact>
    where
        Self: Sized,
        Self: 'a,
    {
        if let Some(first) = fractions.next() {
            let mut cumulative_probabilities = vec![first.clone()];
            let mut highest_denom = first.0.to_denominator();

            while let Some(fraction) = fractions.next() {
                highest_denom = highest_denom.max(fraction.0.to_denominator());

                cumulative_probabilities.push(fraction + cumulative_probabilities.last().unwrap());
            }
            let highest_denom = highest_denom.clone();

            Ok(FractionRandomCacheExact {
                cumulative_probabilities,
                highest_denom,
            })
        } else {
            Err(anyhow!("cannot take an element of an empty list"))
        }
    }

    fn choose_randomly_cached(cache: &FractionRandomCacheExact) -> usize
    where
        Self: Sized,
    {
        //select a random value
        let mut rng = rand::rng();
        let mut buf = [0u8; 32];
        rng.fill_bytes(&mut buf);
        let seed = Seed::from_bytes(buf);
        let rand_val = {
            //strategy: the highest denominator determines how much precision we need

            //Generate a random value with the number of bits of the highest denominator. Repeat until this value is <= the max denominator.
            let rand_val = random_naturals_less_than(seed, cache.highest_denom.clone())
                .next()
                .unwrap();

            //create the fraction from the random nominator and the max denominator
            FractionExact(Rational::from(rand_val) / Rational::from(cache.highest_denom.clone()))
        };

        match cache.cumulative_probabilities.binary_search(&rand_val) {
            Ok(index) | Err(index) => index,
        }
    }
}

pub struct FractionRandomCacheF64 {
    cumulative_probabilities: Vec<FractionF64>,
}

impl ChooseRandomly for FractionF64 {
    type Cache = FractionRandomCacheF64;

    fn choose_randomly(fractions: &Vec<FractionF64>) -> Result<usize> {
        if fractions.is_empty() {
            return Err(anyhow!("cannot take an element of an empty list"));
        }

        //normalise the probabilities
        let mut probabilities: Vec<FractionF64> = fractions.iter().cloned().collect();
        let sum = probabilities
            .iter()
            .fold(FractionF64::zero(), |x, y| &x + y);
        probabilities.retain_mut(|v| {
            *v /= &sum;
            true
        });

        //select a random value
        let mut rng = rand::rng();
        let rand_val = FractionF64(rng.random_range(0.0..=1.0));

        let mut cum_prob = FractionF64::zero();
        for (index, value) in probabilities.iter().enumerate() {
            cum_prob += value;
            if rand_val < cum_prob {
                return Ok(index);
            }
        }
        Ok(probabilities.len() - 1)
    }

    fn choose_randomly_create_cache<'a>(
        mut fractions: impl Iterator<Item = &'a Self>,
    ) -> Result<FractionRandomCacheF64>
    where
        Self: Sized,
        Self: 'a,
    {
        if let Some(first) = fractions.next() {
            let mut cumulative_probabilities = vec![*first];

            while let Some(fraction) = fractions.next() {
                cumulative_probabilities.push(fraction + cumulative_probabilities.last().unwrap());
            }

            Ok(FractionRandomCacheF64 {
                cumulative_probabilities,
            })
        } else {
            Err(anyhow!("cannot take an element of an empty list"))
        }
    }

    fn choose_randomly_cached(cache: &FractionRandomCacheF64) -> usize
    where
        Self: Sized,
    {
        //select a random value
        let mut rng = rand::rng();
        let rand_val = FractionF64::from(
            rng.random_range(
                0.0..=*cache
                    .cumulative_probabilities
                    .last()
                    .unwrap()
                    .approx_ref()
                    .unwrap(),
            ),
        );

        match cache.cumulative_probabilities.binary_search(&rand_val) {
            Ok(index) | Err(index) => index,
        }
    }
}
