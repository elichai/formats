//! Portable branchless backend.
//!
//! Reference implementation for every other backend, the fallback when no SIMD
//! extension is available, and the handler for the tail of each SIMD loop.
//!
//! Based on code from: <https://github.com/Sc00bz/ConstTimeEncoding/blob/master/hex.cpp>

#![deny(unsafe_code)]

use super::{Case, LOWER, MIXED, UPPER};
use crate::Error;

/// Added to a nibble's ASCII value to reach `a-f` rather than `:-?`.
const ALPHA_OFFSET_LOWER: i16 = 0x61 - 0x3a;
/// Added to a nibble's ASCII value to reach `A-F` rather than `:-?`.
const ALPHA_OFFSET_UPPER: i16 = 0x41 - 0x3a;

/// Encodes `src` into `dst`, using the upper-case alphabet if `upper` is set.
///
/// `dst.len()` must be exactly `src.len() * 2`.
#[inline]
pub(crate) fn encode(src: &[u8], dst: &mut [u8], upper: bool) {
    // Selecting the alphabet is a single hoisted value; the loop body is
    // identical for both cases.
    let alpha_offset = if upper {
        ALPHA_OFFSET_UPPER
    } else {
        ALPHA_OFFSET_LOWER
    };

    for (src, dst) in src.iter().zip(dst.chunks_exact_mut(2)) {
        dst[0] = encode_nibble(src >> 4, alpha_offset);
        dst[1] = encode_nibble(src & 0x0f, alpha_offset);
    }
}

/// Decodes `src` into `dst`.
///
/// `src.len()` must be exactly `dst.len() * 2`.
#[inline]
pub(crate) fn decode<const CASE: Case>(src: &[u8], dst: &mut [u8]) -> Result<(), Error> {
    match decode_inner::<CASE>(src, dst) {
        0 => Ok(()),
        _ => Err(Error::InvalidEncoding),
    }
}

/// Decodes `src` into `dst`, returning the raw error accumulator.
///
/// Nonzero iff `src` held a byte outside the alphabet selected by `CASE`. The
/// accumulator is returned rather than a `Result` so that callers chaining onto
/// this backend can defer the single comparison, and so that the constant-time
/// test harness can observe it directly.
#[inline(always)]
pub(crate) fn decode_inner<const CASE: Case>(src: &[u8], dst: &mut [u8]) -> u16 {
    let mut err: u16 = 0;

    for (src, dst) in src.chunks_exact(2).zip(dst.iter_mut()) {
        let byte = (decode_nibble::<CASE>(src[0]) << 4) | decode_nibble::<CASE>(src[1]);
        // Accumulated rather than tested, so an invalid byte costs the same
        // wherever it appears.
        err |= byte >> 8;
        *dst = byte as u8;
    }

    err
}

/// Runtime-`Case` entry point, for SIMD backends dispatching their own tail.
///
/// Keeps the `Case` match in one place rather than repeating it in every
/// backend. `case` is not secret, so the branch has no timing implication.
///
/// Gated to the targets that actually have a SIMD backend, so it does not show
/// up as dead code elsewhere.
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(base16ct_backend = "soft")
))]
#[inline(always)]
pub(crate) fn decode_inner_dyn(src: &[u8], dst: &mut [u8], case: Case) -> u16 {
    match case {
        LOWER => decode_inner::<LOWER>(src, dst),
        UPPER => decode_inner::<UPPER>(src, dst),
        _ => decode_inner::<MIXED>(src, dst),
    }
}

/// Decodes a single nibble, returning a value with the high byte set on error.
///
/// `CASE` is a const parameter, so the `match` is resolved at compile time and
/// each instantiation emits only the range tests its alphabet needs.
///
/// Each range test appears exactly once, as a named term, so that a typo cannot
/// affect one alphabet while leaving the others correct.
#[inline(always)]
fn decode_nibble<const CASE: Case>(src: u8) -> u16 {
    const {
        assert!(CASE <= MIXED, "unknown Case");
    }

    let byte = src as i16;

    // Start at -1 so a byte matching no range leaves the high byte set.
    let ret = -1
        + digit_term(byte)
        + match CASE {
            LOWER => lower_alpha_term(byte),
            UPPER => upper_alpha_term(byte),
            // At most one term is ever nonzero, so summing them is the same as
            // applying whichever one matched.
            _ => upper_alpha_term(byte) + lower_alpha_term(byte),
        };

    ret as u16
}

/// Value of a `0-9` byte (0x30-0x39), or zero for anything else.
///
/// `(lo - byte) & (byte - hi)` is negative exactly inside the range, so the
/// arithmetic shift yields an all-ones or all-zeros mask with no branch.
#[inline(always)]
fn digit_term(byte: i16) -> i16 {
    // if (byte > 0x2f && byte < 0x3a) byte - 0x30 + 1; // -47
    (((0x2fi16 - byte) & (byte - 0x3a)) >> 8) & (byte - 47)
}

/// Value of an `A-F` byte (0x41-0x46), or zero for anything else.
#[inline(always)]
fn upper_alpha_term(byte: i16) -> i16 {
    // if (byte > 0x40 && byte < 0x47) byte - 0x41 + 10 + 1; // -54
    (((0x40i16 - byte) & (byte - 0x47)) >> 8) & (byte - 54)
}

/// Value of an `a-f` byte (0x61-0x66), or zero for anything else.
#[inline(always)]
fn lower_alpha_term(byte: i16) -> i16 {
    // if (byte > 0x60 && byte < 0x67) byte - 0x61 + 10 + 1; // -86
    (((0x60i16 - byte) & (byte - 0x67)) >> 8) & (byte - 86)
}

/// Encodes a single nibble using the alphabet selected by `alpha_offset`.
#[inline(always)]
fn encode_nibble(src: u8, alpha_offset: i16) -> u8 {
    let mut ret = src as i16 + 0x30;
    // 0-9  0x30-0x39, then the alphabetic range for values >= 10
    ret += ((0x39i16 - ret) >> 8) & alpha_offset;
    ret as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::test_support;

    fn exercise<const CASE: Case>() {
        test_support::exercise_backend(encode, decode::<CASE>, CASE);
        test_support::exercise_boundary_sizes(decode::<CASE>, CASE);
    }

    #[test]
    fn lower_matches_reference() {
        exercise::<LOWER>();
        test_support::exercise_case_strictness(decode::<LOWER>, LOWER);
    }

    #[test]
    fn upper_matches_reference() {
        exercise::<UPPER>();
        test_support::exercise_case_strictness(decode::<UPPER>, UPPER);
    }

    #[test]
    fn mixed_matches_reference() {
        exercise::<MIXED>();
    }

    /// The same exhaustive table coverage the SIMD backends get, so the
    /// reference and the accelerated paths are held to one standard.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn decode_all_byte_values() {
        test_support::exercise_all_byte_values(decode::<LOWER>, LOWER);
        test_support::exercise_all_byte_values(decode::<UPPER>, UPPER);
        test_support::exercise_all_byte_values(decode::<MIXED>, MIXED);
    }

    /// Every one of the 256 byte values is accepted or rejected per alphabet.
    ///
    /// Skipped under Miri: 3 x 65536 interpreted iterations is far too slow, and
    /// this backend contains no `unsafe` for Miri to find anything in.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn decode_accepts_exactly_its_alphabet() {
        for case in [LOWER, UPPER, MIXED] {
            for hi in 0..=u8::MAX {
                for lo in 0..=u8::MAX {
                    let src = [hi, lo];
                    let mut dst = [0u8; 1];
                    let mut reference = [0u8; 1];

                    let want = test_support::naive_decode(&src, &mut reference, case);
                    let got = match case {
                        LOWER => decode::<LOWER>(&src, &mut dst),
                        UPPER => decode::<UPPER>(&src, &mut dst),
                        _ => decode::<MIXED>(&src, &mut dst),
                    };

                    assert_eq!(got.is_ok(), want, "{hi:#04x} {lo:#04x} case {case}");
                    if want {
                        assert_eq!(dst, reference, "{hi:#04x} {lo:#04x} case {case}");
                    }
                }
            }
        }
    }
}
