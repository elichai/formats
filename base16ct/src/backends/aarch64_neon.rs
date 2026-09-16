//! AArch64 NEON backend.
//!
//! Compiled only when the `neon` target feature is enabled, which it is by
//! default on `aarch64`, so no runtime detection is involved.
//!
//! # Safety
//!
//! Every NEON intrinsic used here requires the `neon` target feature. The
//! module is `cfg`-gated on `target_feature = "neon"`, so that requirement is
//! satisfied by construction for the whole file; note that enabling the feature
//! in the build configuration does not by itself make the intrinsics safe to
//! call, hence the `unsafe` blocks. The only other obligation is pointer
//! validity, which each block documents against the enclosing bounds check.
//!
//! The entry points stay safe functions, so the dispatch layer needs no
//! `unsafe` on this architecture.
//!
//! # Encoding
//!
//! `vqtbl1q_u8` is a parallel 4-bit table lookup: a register holds the
//! alphabet and nibbles index into it, then `vzipq_u8` interleaves the
//! high-nibble and low-nibble characters. As on x86 the table is in a
//! register, so there is no data-dependent memory access.
//!
//! # Decoding (Muła–Langdale)
//!
//! Two parallel paths merged with `vminq_u8`, which works because the path
//! that does not apply always produces a value >= 16:
//!
//! - digit: `(v + 0xc6) sat_sub 6 - 0xf0` yields `0..=9` for `'0'..='9'`.
//! - alpha: `sat_add(v - base, 10)` yields `10..=15` for the accepted letters.
//!
//! Validation is `sat_add(nibble, 0x70)`: valid nibbles land in `0x70..=0x7f`
//! with the MSB clear, invalid ones saturate past `0x80`.
//!
//! Restricting the alphabet is *cheaper* here rather than free. Mixed case
//! needs a `& 0xdf` fold to map `a-f` onto `A-F`; a single-case decoder skips
//! the fold and uses its own `base`, because the wrong case then lands outside
//! the range by wrapping. `'A'` against `base = b'a'` gives
//! `0x41 - 0x61 = 0xe0`, and `sat_add(0xe0, 10) = 0xea`, far past 16. Since
//! `CASE` is a const parameter the fold is eliminated entirely for `lower` and
//! `upper`.

#![allow(unsafe_code)]

use core::arch::aarch64::*;

use super::{Case, LOWER, MIXED, soft};
use crate::Error;

/// Alphabet table for `vqtbl1q_u8`, held in a register rather than memory.
#[inline(always)]
fn hex_lut(upper: bool) -> uint8x16_t {
    let bytes: [u8; 16] = if upper {
        *b"0123456789ABCDEF"
    } else {
        *b"0123456789abcdef"
    };

    // SAFETY: `neon` is guaranteed by the module gate; reads 16 bytes from a
    // 16-byte array.
    unsafe { vld1q_u8(bytes.as_ptr()) }
}

/// Encodes one 16-byte chunk at `src + offset` into `dst + offset * 2`.
///
/// # Safety
///
/// `offset + 16 <= src.len()`, and `dst.len()` must be `src.len() * 2`.
#[inline(always)]
unsafe fn encode_chunk(src: &[u8], dst: &mut [u8], offset: usize, lut: uint8x16_t) {
    // `offset` is a parameter, so assert the precondition rather than trusting
    // every call site. This is worth having because an off-by-one here reads
    // and writes out of bounds while leaving every in-bounds output byte
    // correct, so comparing output against a reference would not notice; only
    // Miri or this assertion catches it.
    debug_assert!(offset + 16 <= src.len());
    debug_assert!(offset * 2 + 32 <= dst.len());

    // SAFETY: `neon` is guaranteed by the module gate, and the caller
    // guarantees 16 readable bytes at `offset` and 32 writable at `offset * 2`.
    unsafe {
        let mask_lo = vdupq_n_u8(0x0f);
        let chunk = vld1q_u8(src.as_ptr().add(offset));

        let hi = vshrq_n_u8::<4>(chunk);
        let lo = vandq_u8(chunk, mask_lo);
        let zipped = vzipq_u8(vqtbl1q_u8(lut, hi), vqtbl1q_u8(lut, lo));

        let out = dst.as_mut_ptr().add(offset * 2);
        vst1q_u8(out, zipped.0);
        vst1q_u8(out.add(16), zipped.1);
    }
}

/// Encodes `src` into `dst`, using the upper-case alphabet if `upper` is set.
///
/// `dst.len()` must be exactly `src.len() * 2`.
#[inline(always)]
pub(crate) fn encode_inner(src: &[u8], dst: &mut [u8], upper: bool) {
    debug_assert_eq!(dst.len(), src.len() * 2);

    let lut = hex_lut(upper);

    // Two chunks per iteration, to give the out-of-order engine more
    // independent work than a single dependent chain would.
    let mut done = 0;
    while src.len() - done >= 32 {
        // SAFETY: `done + 32 <= src.len()`, so both chunks are in bounds.
        unsafe {
            encode_chunk(src, dst, done, lut);
            encode_chunk(src, dst, done + 16, lut);
        }
        done += 32;
    }

    if src.len() - done >= 16 {
        // SAFETY: `done + 16 <= src.len()`.
        unsafe {
            encode_chunk(src, dst, done, lut);
        }
        done += 16;
    }

    if done < src.len() {
        soft::encode(&src[done..], &mut dst[done * 2..], upper);
    }
}

/// Decodes 16 hex characters to nibble values, `>= 16` marking invalid input.
#[inline(always)]
fn decode_nibbles<const CASE: Case>(v: uint8x16_t) -> uint8x16_t {
    // SAFETY: `neon` is guaranteed by the module gate; no memory is touched.
    unsafe {
        // Digit path: maps `'0'..='9'` to `0..=9`.
        let digit = vsubq_u8(
            vqsubq_u8(vaddq_u8(v, vdupq_n_u8(0xc6)), vdupq_n_u8(6)),
            vdupq_n_u8(0xf0),
        );

        // Alpha path. Mixed case folds `a-f` onto `A-F`; a single-case decoder
        // skips the fold, which is what makes the other case land out of range.
        let folded = if CASE == MIXED {
            vandq_u8(v, vdupq_n_u8(0xdf))
        } else {
            v
        };
        let base = if CASE == LOWER { b'a' } else { b'A' };
        let alpha = vqaddq_u8(vsubq_u8(folded, vdupq_n_u8(base)), vdupq_n_u8(10));

        // The inapplicable path always exceeds 15, so `min` picks the right one.
        vminq_u8(digit, alpha)
    }
}

/// Decodes `src` into `dst`, returning the raw error accumulator.
///
/// `src.len()` must be exactly `dst.len() * 2`.
#[inline(always)]
pub(crate) fn decode_inner<const CASE: Case>(src: &[u8], dst: &mut [u8]) -> u8 {
    const {
        assert!(CASE <= MIXED, "unknown Case");
    }
    debug_assert_eq!(src.len(), dst.len() * 2);

    // SAFETY: `neon` is guaranteed by the module gate. Each load and store is
    // covered by the `dst.len() - done >= 16` condition, which bounds 16 bytes
    // of output and, since `src` is twice as long, 32 bytes of input.
    let (mut err, done) = unsafe {
        let validate = vdupq_n_u8(0x70);

        // OR-accumulate the per-iteration check vectors and reduce once on
        // exit, replacing a cross-lane `vmaxvq_u8` per iteration with a single
        // `orr`.
        //
        // Do *not* mirror this on x86: there `pmovmskb` runs on a port
        // disjoint from the SIMD ALU on Zen, and the same transform regresses.
        let mut acc = vdupq_n_u8(0);

        // 32 hex characters -> 16 bytes per iteration.
        let mut done = 0;
        while dst.len() - done >= 16 {
            let hex = src.as_ptr().add(done * 2);
            let v0 = vld1q_u8(hex);
            let v1 = vld1q_u8(hex.add(16));

            let nib0 = decode_nibbles::<CASE>(v0);
            let nib1 = decode_nibbles::<CASE>(v1);

            acc = vorrq_u8(
                acc,
                vorrq_u8(vqaddq_u8(nib0, validate), vqaddq_u8(nib1, validate)),
            );

            // Deinterleave the nibble pairs, then combine into bytes.
            let split = vuzpq_u8(nib0, nib1);
            let combined = vorrq_u8(vshlq_n_u8::<4>(split.0), split.1);
            vst1q_u8(dst.as_mut_ptr().add(done), combined);

            done += 16;
        }

        (vmaxvq_u8(acc) & 0x80, done)
    };

    if done < dst.len() {
        let tail = soft::decode_inner::<CASE>(&src[done * 2..], &mut dst[done..]);
        // Fold both halves before narrowing. `soft`'s accumulator happens to
        // keep its bits in the low byte, but relying on that would make a
        // silent truncation load-bearing; this is branchless either way.
        err |= (tail | (tail >> 8)) as u8;
    }

    err
}

/// Encodes `src` into `dst`, using the upper-case alphabet if `upper` is set.
///
/// `dst.len()` must be exactly `src.len() * 2`.
#[inline]
pub(crate) fn encode(src: &[u8], dst: &mut [u8], upper: bool) {
    encode_inner(src, dst, upper);
}

/// Decodes `src` into `dst`, accepting only the alphabet selected by `CASE`.
///
/// `src.len()` must be exactly `dst.len() * 2`.
#[inline]
pub(crate) fn decode<const CASE: Case>(src: &[u8], dst: &mut [u8]) -> Result<(), Error> {
    match decode_inner::<CASE>(src, dst) {
        0 => Ok(()),
        _ => Err(Error::InvalidEncoding),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::{UPPER, test_support};

    /// NEON is a compile-time guarantee here, so unlike the x86 tiers there is
    /// no availability check and no way for these to skip.
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
