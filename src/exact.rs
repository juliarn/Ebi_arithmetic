//======================== exactness tools ========================//

use anyhow::Result;
use std::sync::atomic::AtomicBool;

static EXACT: AtomicBool = AtomicBool::new(true);

/// Enables or disables exact arithmetic globally.
/// Exact arithmetic cannot be combined with approximate arithmetic.
pub fn set_exact_globally(exact: bool) {
    EXACT.store(exact, std::sync::atomic::Ordering::Relaxed);
}

pub fn is_exact_globally() -> bool {
    if cfg!(any(
        all(
            feature = "exactarithmetic",
            feature = "approximatearithmetic"
        ),
        all(
            not(feature = "exactarithmetic"),
            not(feature = "approximatearithmetic")
        )
    )) {
        EXACT.load(std::sync::atomic::Ordering::Relaxed)
    } else if cfg!(feature = "exactarithmetic") {
        true
    } else {
        false
    }
}

pub trait MaybeExact {
    type Approximate;
    type Exact;

    fn is_exact(&self) -> bool;

    /**
     * This is a low-level function to extract an approximate value. Will only succeed if the fraction is approximate.
     */
    fn approx_ref(&self) -> Result<&Self::Approximate>;

    /**
     * This is a low-level function to extract an exact value. Will only succeed if the fraction is exact.
     */
    fn exact_ref(&self) -> Result<&Self::Exact>;

    fn approx(self) -> Result<Self::Approximate>;

    fn exact(self) -> Result<Self::Exact>;
}
