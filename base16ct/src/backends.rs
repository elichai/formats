//! Encoding and decoding backends.
//!
//! The backend is chosen at runtime by default: on x86 the widest supported
//! tier wins, elsewhere `soft` is used. It can be pinned at compile time with
//! `--cfg base16ct_backend="..."`; see the crate README for the values.
//!
//! A pinned backend is not a separate code path. Pinning excludes the wider
//! tiers from the build and requires the matching `target_feature`, and because
//! `cpufeatures` detection folds to a constant when the feature is already
//! enabled, the cascade below collapses to the pinned call with everything else
//! eliminated.

use crate::Error;

/// Hex alphabet accepted by a decoder.
///
/// A `u8` rather than an `enum` because only primitives can be const generic
/// parameters on stable Rust.
pub(crate) type Case = u8;

/// Accept `0-9a-f` only.
pub(crate) const LOWER: Case = 0;
/// Accept `0-9A-F` only.
pub(crate) const UPPER: Case = 1;
/// Accept `0-9a-fA-F`.
pub(crate) const MIXED: Case = 2;

pub(crate) mod soft;

/// True when x86 SIMD backends are compiled in at all.
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(base16ct_backend = "soft")
))]
mod x86_ssse3;

/// AVX2 is excluded when a lower tier is pinned.
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(base16ct_backend = "soft"),
    not(base16ct_backend = "x86-ssse3")
))]
mod x86_avx2;

#[cfg(all(base16ct_backend = "x86-ssse3", not(target_feature = "ssse3")))]
compile_error!(r#"base16ct_backend="x86-ssse3" requires the `ssse3` target feature"#);

#[cfg(all(base16ct_backend = "x86-avx2", not(target_feature = "avx2")))]
compile_error!(r#"base16ct_backend="x86-avx2" requires the `avx2` target feature"#);

#[cfg(all(
    any(base16ct_backend = "x86-ssse3", base16ct_backend = "x86-avx2"),
    not(any(target_arch = "x86", target_arch = "x86_64"))
))]
compile_error!("the pinned base16ct backend is only available on x86 targets");

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(base16ct_backend = "soft"),
    not(base16ct_backend = "x86-ssse3")
))]
cpufeatures::new!(avx2_cpuid, "avx2");

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(base16ct_backend = "soft")
))]
cpufeatures::new!(ssse3_cpuid, "ssse3");

/// Encodes `src` into `dst`, using the upper-case alphabet if `upper` is set.
///
/// `dst.len()` must be exactly `src.len() * 2`.
// Every `unsafe` call below is guarded by its own CPU feature detection.
#[allow(unsafe_code)]
#[inline]
pub(crate) fn encode(src: &[u8], dst: &mut [u8], upper: bool) {
    debug_assert_eq!(dst.len(), src.len() * 2);

    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        not(base16ct_backend = "soft")
    ))]
    {
        #[cfg(not(base16ct_backend = "x86-ssse3"))]
        if avx2_cpuid::get() {
            // SAFETY: AVX2 support was just confirmed, and the length
            // precondition is asserted above.
            return unsafe { x86_avx2::encode(src, dst, upper) };
        }

        if ssse3_cpuid::get() {
            // SAFETY: SSSE3 support was just confirmed, and the length
            // precondition is asserted above.
            return unsafe { x86_ssse3::encode(src, dst, upper) };
        }
    }

    soft::encode(src, dst, upper);
}

/// Decodes `src` into `dst`.
///
/// `src.len()` must be exactly `dst.len() * 2`.
///
/// On error the contents of `dst` are unspecified: every byte is processed
/// regardless of validity, so that the time taken does not depend on *where* an
/// invalid byte occurs.
// Every `unsafe` call below is guarded by its own CPU feature detection.
#[allow(unsafe_code)]
#[inline]
pub(crate) fn decode<const CASE: Case>(src: &[u8], dst: &mut [u8]) -> Result<(), Error> {
    debug_assert_eq!(src.len(), dst.len() * 2);

    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        not(base16ct_backend = "soft")
    ))]
    {
        // `CASE` is erased to a runtime argument here: the x86 tiers select
        // their `pshufb` tables before the loop, so one compiled copy serves
        // all three alphabets instead of three.
        #[cfg(not(base16ct_backend = "x86-ssse3"))]
        if avx2_cpuid::get() {
            // SAFETY: AVX2 support was just confirmed, and the length
            // precondition is asserted above.
            return unsafe { x86_avx2::decode(src, dst, CASE) };
        }

        if ssse3_cpuid::get() {
            // SAFETY: SSSE3 support was just confirmed, and the length
            // precondition is asserted above.
            return unsafe { x86_ssse3::decode(src, dst, CASE) };
        }
    }

    // `CASE` stays static here: `soft`'s inner match folds to one path.
    soft::decode::<CASE>(src, dst)
}

#[cfg(test)]
pub(crate) mod test_support;

#[cfg(test)]
mod tests {
    use super::*;

    /// Exercises whichever backend dispatch actually selects on this host.
    ///
    /// The per-backend tests skip when their CPU feature is unavailable, so in
    /// principle every one of them could skip. This test cannot: it always
    /// runs, and always covers the tier that real callers will get.
    fn run<const CASE: Case>() {
        test_support::exercise_backend(encode, decode::<CASE>, CASE);
        test_support::exercise_boundary_sizes(decode::<CASE>, CASE);

        if CASE != MIXED {
            test_support::exercise_case_strictness(decode::<CASE>, CASE);
        }
    }

    #[test]
    fn dispatch_lower_matches_reference() {
        run::<LOWER>();
    }

    #[test]
    fn dispatch_upper_matches_reference() {
        run::<UPPER>();
    }

    #[test]
    fn dispatch_mixed_matches_reference() {
        run::<MIXED>();
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn dispatch_decode_all_byte_values() {
        test_support::exercise_all_byte_values(decode::<LOWER>, LOWER);
        test_support::exercise_all_byte_values(decode::<UPPER>, UPPER);
        test_support::exercise_all_byte_values(decode::<MIXED>, MIXED);
    }
}
