//! Encoding and decoding backends.
//!
//! The backend is selected at runtime. It can be pinned at compile time with
//! `--cfg base16ct_backend="..."`; see the crate README for the supported
//! values.

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

/// Encodes `src` into `dst`, using the upper-case alphabet if `upper` is set.
///
/// `dst.len()` must be exactly `src.len() * 2`.
#[inline]
pub(crate) fn encode(src: &[u8], dst: &mut [u8], upper: bool) {
    debug_assert_eq!(dst.len(), src.len() * 2);
    soft::encode(src, dst, upper);
}

/// Decodes `src` into `dst`.
///
/// `src.len()` must be exactly `dst.len() * 2`.
///
/// On error the contents of `dst` are unspecified: every byte is processed
/// regardless of validity, so that the time taken does not depend on *where*
/// an invalid byte occurs.
#[inline]
pub(crate) fn decode<const CASE: Case>(src: &[u8], dst: &mut [u8]) -> Result<(), Error> {
    debug_assert_eq!(src.len(), dst.len() * 2);
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
