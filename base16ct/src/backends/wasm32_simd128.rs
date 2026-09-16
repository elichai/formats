//! WebAssembly `simd128` backend.
//!
//! Compiled only when the `simd128` target feature is enabled, so there is no
//! runtime detection. WebAssembly has no equivalent of CPUID: whether the
//! runtime supports SIMD is a property of the module as loaded, which is why
//! this is a compile-time choice rather than a dispatch.
//!
//! # Safety
//!
//! Only the load and store intrinsics are unsafe here, and each is covered by
//! the enclosing bounds check. Unlike the x86 and NEON backends, the lane
//! arithmetic needs no `unsafe`.
//!
//! # Encoding
//!
//! `u8x16_swizzle` is a parallel 4-bit lookup with the alphabet held in a
//! register, then `u8x16_shuffle` interleaves the high-nibble and low-nibble
//! characters. As on the other backends the table is in a register, so there
//! is no data-dependent memory access.
//!
//! # Decoding (Muła–Langdale)
//!
//! Identical in structure to [`super::aarch64_neon`]: a digit path and an
//! alpha path merged with `u8x16_min`, validated by saturating addition of
//! `112`, and restricted to a single alphabet by skipping the `& 0xdf` fold.
//! See that module for why skipping the fold rejects the other case.

use core::arch::wasm32::*;

use super::{Case, LOWER, MIXED, soft};
use crate::Error;

/// Alphabet table for `u8x16_swizzle`, held in a register rather than memory.
#[inline(always)]
fn hex_lut(upper: bool) -> v128 {
    let table = if upper {
        *b"0123456789ABCDEF"
    } else {
        *b"0123456789abcdef"
    };

    // SAFETY: reads 16 bytes from a 16-byte array.
    #[allow(unsafe_code)]
    unsafe {
        v128_load(table.as_ptr().cast())
    }
}

/// Encodes `src` into `dst`, 16 input bytes per iteration.
///
/// `dst.len()` must be exactly `src.len() * 2`.
#[inline(always)]
pub(super) fn encode_inner(src: &[u8], dst: &mut [u8], upper: bool) {
    debug_assert_eq!(dst.len(), src.len() * 2);

    let lut = hex_lut(upper);
    let mask_lo = u8x16_splat(0x0f);

    let mut done = 0;
    while src.len() - done >= 16 {
        // SAFETY: `done + 16 <= src.len()`, so the 16-byte read is in bounds,
        // and `dst` is twice as long so the two 16-byte writes are too.
        #[allow(unsafe_code)]
        unsafe {
            let chunk = v128_load(src.as_ptr().add(done).cast());

            let lo = v128_and(chunk, mask_lo);
            let hi = u8x16_shr(chunk, 4);

            let lo_ascii = u8x16_swizzle(lut, lo);
            let hi_ascii = u8x16_swizzle(lut, hi);

            // High-nibble character first, then low, per input byte.
            let out0 = u8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(
                hi_ascii, lo_ascii,
            );
            let out1 = u8x16_shuffle::<8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31>(
                hi_ascii, lo_ascii,
            );

            let out = dst.as_mut_ptr().add(done * 2);
            v128_store(out.cast(), out0);
            v128_store(out.add(16).cast(), out1);
        }

        done += 16;
    }

    if done < src.len() {
        soft::encode(&src[done..], &mut dst[done * 2..], upper);
    }
}

/// Decodes 16 hex characters to nibble values, returning the values and the
/// MSB-per-lane error bitmask.
#[inline(always)]
fn decode_nibbles<const CASE: Case>(v: v128) -> (v128, u16) {
    // Digit path: maps `'0'..='9'` to `0..=9`.
    let digit = u8x16_sub(
        u8x16_sub_sat(u8x16_add(v, u8x16_splat(0xc6)), u8x16_splat(6)),
        u8x16_splat(0xf0),
    );

    // Alpha path. Mixed case folds `a-f` onto `A-F`; a single-case decoder
    // skips the fold, which is what makes the other case land out of range.
    let folded = if CASE == MIXED {
        v128_and(v, u8x16_splat(0xdf))
    } else {
        v
    };
    let base = if CASE == LOWER { b'a' } else { b'A' };
    let alpha = u8x16_add_sat(u8x16_sub(folded, u8x16_splat(base)), u8x16_splat(10));

    // The inapplicable path always exceeds 15, so `min` picks the right one.
    let nibbles = u8x16_min(digit, alpha);

    // Valid nibbles stay <= 127 after adding 112; anything >= 16 saturates
    // past 128, so the per-lane MSB is the error flag.
    let err = u8x16_bitmask(u8x16_add_sat(nibbles, u8x16_splat(112)));

    (nibbles, err)
}

/// Decodes `src` into `dst`, returning the raw error accumulator.
///
/// `src.len()` must be exactly `dst.len() * 2`.
#[inline(always)]
pub(super) fn decode_inner<const CASE: Case>(src: &[u8], dst: &mut [u8]) -> u16 {
    const {
        assert!(CASE <= MIXED, "unknown Case");
    }
    debug_assert_eq!(src.len(), dst.len() * 2);

    let mut err: u16 = 0;

    // 32 hex characters -> 16 bytes per iteration.
    let mut done = 0;
    while dst.len() - done >= 16 {
        // SAFETY: `done + 16 <= dst.len()` and `src` is twice as long, so both
        // 16-byte reads are in bounds.
        #[allow(unsafe_code)]
        let (v0, v1) = unsafe {
            let hex = src.as_ptr().add(done * 2);
            (v128_load(hex.cast()), v128_load(hex.add(16).cast()))
        };

        let (nib0, err0) = decode_nibbles::<CASE>(v0);
        let (nib1, err1) = decode_nibbles::<CASE>(v1);
        err |= err0 | err1;

        // Deinterleave the nibble pairs, then combine into bytes.
        let hi =
            u8x16_shuffle::<0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30>(nib0, nib1);
        let lo =
            u8x16_shuffle::<1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31>(nib0, nib1);
        let packed = v128_or(u8x16_shl(hi, 4), lo);

        // SAFETY: as above; `dst` has 16 writable bytes at `done`.
        #[allow(unsafe_code)]
        unsafe {
            v128_store(dst.as_mut_ptr().add(done).cast(), packed);
        }

        done += 16;
    }

    if done < dst.len() {
        err |= soft::decode_inner::<CASE>(&src[done * 2..], &mut dst[done..]);
    }

    err
}

/// Encodes `src` into `dst`, using the upper-case alphabet if `upper` is set.
///
/// `dst.len()` must be exactly `src.len() * 2`.
#[inline]
pub(super) fn encode(src: &[u8], dst: &mut [u8], upper: bool) {
    encode_inner(src, dst, upper);
}

/// Decodes `src` into `dst`, accepting only the alphabet selected by `CASE`.
///
/// `src.len()` must be exactly `dst.len() * 2`.
#[inline]
pub(super) fn decode<const CASE: Case>(src: &[u8], dst: &mut [u8]) -> Result<(), Error> {
    match decode_inner::<CASE>(src, dst) {
        0 => Ok(()),
        _ => Err(Error::InvalidEncoding),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::{UPPER, test_support};

    /// `simd128` is a compile-time guarantee here, so as with NEON there is no
    /// availability check and no way for these to skip.
    fn run<const CASE: Case>() {
        test_support::exercise_backend(encode, decode::<CASE>, CASE);
        test_support::exercise_boundary_sizes(decode::<CASE>, CASE);

        if CASE != MIXED {
            test_support::exercise_case_strictness(decode::<CASE>, CASE);
        }
    }

    #[test]
    fn lower_matches_reference() {
        run::<LOWER>();
    }

    #[test]
    fn upper_matches_reference() {
        run::<UPPER>();
    }

    #[test]
    fn mixed_matches_reference() {
        run::<MIXED>();
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn decode_accepts_exactly_its_alphabet() {
        test_support::exercise_all_byte_values(decode::<LOWER>, LOWER);
        test_support::exercise_all_byte_values(decode::<UPPER>, UPPER);
        test_support::exercise_all_byte_values(decode::<MIXED>, MIXED);
    }
}
