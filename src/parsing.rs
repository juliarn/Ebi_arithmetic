use anyhow::Error;
use std::str::FromStr;

use crate::fraction::{fraction_enum::FractionEnum, fraction_exact::FractionExact, fraction_f64::FractionF64};

#[derive(Clone)]
pub struct FractionNotParsedYet {
    pub s: String,
}

impl FromStr for FractionNotParsedYet {
    type Err = Error;

    fn from_str(s: &str) -> std::prelude::v1::Result<Self, Self::Err> {
        Ok(Self { s: s.to_string() })
    }
}

impl TryFrom<&FractionNotParsedYet> for FractionEnum {
    type Error = Error;

    fn try_from(value: &FractionNotParsedYet) -> std::result::Result<Self, Self::Error> {
        Self::from_str(&value.s)
    }
}

impl TryFrom<&FractionNotParsedYet> for FractionExact {
    type Error = Error;

    fn try_from(value: &FractionNotParsedYet) -> std::result::Result<Self, Self::Error> {
        Self::from_str(&value.s)
    }
}

impl TryFrom<&FractionNotParsedYet> for FractionF64 {
    type Error = Error;

    fn try_from(value: &FractionNotParsedYet) -> std::result::Result<Self, Self::Error> {
        Ok(Self::from_str(&value.s)?)
    }
}
