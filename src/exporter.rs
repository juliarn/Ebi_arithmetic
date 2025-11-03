use crate::{
    MaybeExact,
    fraction::{
        fraction_enum::FractionEnum, fraction_exact::FractionExact, fraction_f64::FractionF64,
    },
};
use anyhow::Result;
use malachite::rational::Rational;

pub trait Exporter {
    fn export(&self, f: &mut dyn std::io::Write) -> Result<()>;
}

impl Exporter for String {
    fn export(&self, f: &mut dyn std::io::Write) -> Result<()> {
        Ok(writeln!(f, "{}", self)?)
    }
}

impl Exporter for usize {
    fn export(&self, f: &mut dyn std::io::Write) -> Result<()> {
        Ok(writeln!(f, "{}", self)?)
    }
}

impl Exporter for bool {
    fn export(&self, f: &mut dyn std::io::Write) -> Result<()> {
        Ok(writeln!(f, "{}", self)?)
    }
}

impl Exporter for Rational {
    fn export(&self, f: &mut dyn std::io::Write) -> Result<()> {
        writeln!(f, "{}", self)?;
        Ok(writeln!(f, "Approximately {:.4}", self)?)
    }
}

impl Exporter for f64 {
    fn export(&self, f: &mut dyn std::io::Write) -> Result<()> {
        Ok(writeln!(f, "Approximately {}", self)?)
    }
}

impl Exporter for FractionExact {
    fn export(&self, f: &mut dyn std::io::Write) -> Result<()> {
        self.exact_ref().unwrap().export(f)
    }
}

impl Exporter for FractionF64 {
    fn export(&self, f: &mut dyn std::io::Write) -> Result<()> {
        self.approx_ref().unwrap().export(f)
    }
}

impl Exporter for FractionEnum {
    fn export(&self, f: &mut dyn std::io::Write) -> Result<()> {
        match self {
            FractionEnum::Exact(r) => r.export(f),
            FractionEnum::Approx(r) => r.export(f),
            FractionEnum::CannotCombineExactAndApprox => Ok(write!(
                f,
                "cannot combine exact and approximate arithmetic"
            )?),
        }
    }
}
