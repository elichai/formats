//! SSSE3 backend.
//!
//! # Encoding
//!
//! `pshufb` (`_mm_shuffle_epi8`) is used as a parallel 4-bit lookup: a 16-byte
//! register holds the alphabet (`0-9a-f` or `0-9A-F`) and nibbles extracted
//! from the input serve as shuffle indices. 16 input bytes become 32 hex
//! characters per iteration.
//!
//! The table lives in a **register**, not memory, so unlike a conventional
//! lookup table it introduces no data-dependent memory access and therefore no
//! cache-timing channel.
//!
//! # Decoding (Lemire 2023)
//!
//! Based on <https://lemire.me/blog/2023/12/22/fast-hexadecimal-decoding/>.
//!
//! Subtract 1 from each byte, then use the high nibble of the result as a hash
//! key into two `pshufb` tables:
//!
//! - `delta_check` adds a bias leaving valid characters MSB-clear and invalid
//!   ones MSB-set, so `pmovmskb` collects the errors.
//! - `delta_rebase` adds a bias converting the character to its nibble value.
//!
//! `pmaddubsw` with `{1, 16}` then packs adjacent nibble pairs into bytes and
//! `packuswb` narrows back to `u8`.

// This module is intrinsics throughout; `unsafe` is the point of it.
#![allow(unsafe_code)]
#![allow(unsafe_op_in_unsafe_fn)]

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use super::{Case, LOWER, UPPER, soft};
use crate::Error;

/// Alphabet table for `pshufb`, held in a register rather than memory.
#[inline(always)]
pub(super) unsafe fn hex_lut_128(upper: bool) -> __m128i {
    let lo = i64::from_le_bytes(*b"01234567");
    let hi = if upper {
        i64::from_le_bytes(*b"89ABCDEF")
    } else {
        i64::from_le_bytes(*b"89abcdef")
    };
    _mm_set_epi64x(hi, lo)
}

/// Bias that leaves every byte hashing into a bucket MSB-set, i.e. rejected.
///
/// Buckets 4 and 6 hold `vm1` values `0x40..=0x4f` and `0x60..=0x6f`; adding
/// `-128` puts both ranges in `0x80..=0xff`, so the MSB is set throughout.
const REJECT: i8 = -128;

/// `delta_check` table for the Lemire decode, restricted to `case`.
///
/// Adding the looked-up delta to `vm1 = byte - 1` leaves valid characters
/// MSB-clear and everything else MSB-set.
///
/// Restricting the alphabet is a **table change only**: uppercase `A-F` hashes
/// into bucket 4 and lowercase `a-f` into bucket 6, so rejecting one case costs
/// no extra instructions in the loop.
#[inline(always)]
pub(super) unsafe fn delta_check_128(case: Case) -> __m128i {
    let upper_bucket = if case == LOWER { REJECT } else { 58 };
    let lower_bucket = if case == UPPER { REJECT } else { 26 };

    _mm_setr_epi8(
        -16,          // bucket 0 - no valid character hashes here
        -32,          // bucket 1 - no valid character hashes here
        -47,          // bucket 2 - digit '0' alone (vm1 = 0x2f)
        71,           // bucket 3 - digits '1'-'9' (vm1 = 0x30..=0x38)
        upper_bucket, // bucket 4 - uppercase 'A'-'F' (vm1 = 0x40..=0x45)
        -96,          // bucket 5 - no valid character hashes here
        lower_bucket, // bucket 6 - lowercase 'a'-'f' (vm1 = 0x60..=0x65)
        -128,         // bucket 7 - no valid character hashes here
        // Buckets 8-15 receive bytes with `vm1` high nibble >= 8, i.e. input
        // bytes >= 0x81, all invalid. A zero delta leaves `check = vm1`, whose
        // MSB is already set there, so rejection is automatic.
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )
}

/// `delta_rebase` table for the Lemire decode.
///
/// Adding the looked-up delta to `vm1` yields the nibble value for a valid
/// character. This table is **not** restricted by case: when `delta_check`
/// rejects a bucket the corresponding output byte is unspecified anyway, so
/// leaving the rebase entry in place keeps one fewer value to select.
#[inline(always)]
pub(super) unsafe fn delta_rebase_128() -> __m128i {
    _mm_setr_epi8(
        0,
        0,
        -48 + 1, // bucket 2: digit '0' alone
        -48 + 1, // bucket 3: digits '1'-'9'
        -55 + 1, // bucket 4: uppercase 'A'-'F'
        0,
        -87 + 1, // bucket 6: lowercase 'a'-'f'
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )
}

/// Decodes 16 hex characters into 8 bytes, returning `(packed, check)`.
///
/// `check` has the MSB set in each lane whose input byte was invalid. Callers
/// OR several `check` vectors together and do a single `pmovmskb`, which halves
/// the number of horizontal reductions per iteration.
#[inline(always)]
pub(crate) unsafe fn decode_chunk_128(
    chunk: __m128i,
    delta_check: __m128i,
    delta_rebase: __m128i,
    one: __m128i,
    mask_lo: __m128i,
    weights: __m128i,
) -> (__m128i, __m128i) {
    let vm1 = _mm_sub_epi8(chunk, one);
    let hash_key = _mm_and_si128(_mm_srli_epi16(vm1, 4), mask_lo);

    let check = _mm_add_epi8(vm1, _mm_shuffle_epi8(delta_check, hash_key));
    let nibbles = _mm_add_epi8(vm1, _mm_shuffle_epi8(delta_rebase, hash_key));

    // hi * 16 + lo, then narrow the i16 lanes back to u8.
    let packed16 = _mm_maddubs_epi16(nibbles, weights);
    let packed8 = _mm_packus_epi16(packed16, packed16);

    (packed8, check)
}

/// Encodes `src` into `dst`, 16 input bytes per iteration.
///
/// Carries no `#[target_feature]` so that a higher tier can inline it for its
/// own tail without crossing a call boundary.
///
/// # Safety
///
/// - The CPU must support SSSE3 (the caller must carry
///   `#[target_feature(enable = "ssse3")]`).
/// - `dst.len()` must be exactly `src.len() * 2`.
#[inline(always)]
pub(crate) unsafe fn encode_inner(src: &[u8], dst: &mut [u8], upper: bool) {
    debug_assert_eq!(dst.len(), src.len() * 2);

    let lut = hex_lut_128(upper);
    let mask_lo = _mm_set1_epi8(0x0f);

    let mut done = 0;
    while src.len() - done >= 16 {
        // SAFETY: `done + 16 <= src.len()`, so the 16-byte read is in bounds,
        // and `dst` is twice as long so the two 16-byte writes are too.
        let chunk = _mm_loadu_si128(src.as_ptr().add(done).cast());

        let lo = _mm_and_si128(chunk, mask_lo);
        let hi = _mm_and_si128(_mm_srli_epi16(chunk, 4), mask_lo);

        let hex_lo = _mm_shuffle_epi8(lut, lo);
        let hex_hi = _mm_shuffle_epi8(lut, hi);

        // High-nibble character first, then low.
        let out0 = _mm_unpacklo_epi8(hex_hi, hex_lo);
        let out1 = _mm_unpackhi_epi8(hex_hi, hex_lo);

        let out = dst.as_mut_ptr().add(done * 2);
        _mm_storeu_si128(out.cast(), out0);
        _mm_storeu_si128(out.add(16).cast(), out1);

        done += 16;
    }

    if done < src.len() {
        soft::encode(&src[done..], &mut dst[done * 2..], upper);
    }
}

/// Decodes `src` into `dst`, returning the raw error accumulator.
///
/// Every chunk is processed regardless of validity; errors are OR-accumulated
/// and never branched on, so the time taken does not depend on where an invalid
/// byte occurs.
///
/// # Safety
///
/// - The CPU must support SSSE3 (the caller must carry
///   `#[target_feature(enable = "ssse3")]`).
/// - `src.len()` must be exactly `dst.len() * 2`.
#[inline(always)]
pub(crate) unsafe fn decode_inner(src: &[u8], dst: &mut [u8], case: Case) -> i32 {
    debug_assert_eq!(src.len(), dst.len() * 2);

    let delta_check = delta_check_128(case);
    let delta_rebase = delta_rebase_128();
    let one = _mm_set1_epi8(1);
    let mask_lo = _mm_set1_epi8(0x0f);
    let weights = _mm_set1_epi16(0x0110);
    let mut err = 0i32;

    // 32 hex characters -> 16 bytes per iteration.
    let mut done = 0;
    while dst.len() - done >= 16 {
        // SAFETY: `done + 16 <= dst.len()` and `src` is twice as long, so both
        // 16-byte reads and both 8-byte writes are in bounds.
        let hex = src.as_ptr().add(done * 2);
        let chunk0 = _mm_loadu_si128(hex.cast());
        let chunk1 = _mm_loadu_si128(hex.add(16).cast());

        let (decoded0, check0) =
            decode_chunk_128(chunk0, delta_check, delta_rebase, one, mask_lo, weights);
        let (decoded1, check1) =
            decode_chunk_128(chunk1, delta_check, delta_rebase, one, mask_lo, weights);

        err |= _mm_movemask_epi8(_mm_or_si128(check0, check1));

        let out = dst.as_mut_ptr().add(done);
        _mm_storel_epi64(out.cast(), decoded0);
        _mm_storel_epi64(out.add(8).cast(), decoded1);

        done += 16;
    }

    if done < dst.len() {
        err |= i32::from(soft::decode_inner_dyn(
            &src[done * 2..],
            &mut dst[done..],
            case,
        ));
    }

    err
}

/// Encodes `src` into `dst` using the upper-case alphabet if `upper` is set.
///
/// # Safety
///
/// - The CPU must support SSSE3.
/// - `dst.len()` must be exactly `src.len() * 2`.
#[target_feature(enable = "ssse3")]
pub(crate) unsafe fn encode(src: &[u8], dst: &mut [u8], upper: bool) {
    encode_inner(src, dst, upper)
}

/// Decodes `src` into `dst`, accepting only the alphabet selected by `case`.
///
/// # Safety
///
/// - The CPU must support SSSE3.
/// - `src.len()` must be exactly `dst.len() * 2`.
#[target_feature(enable = "ssse3")]
pub(crate) unsafe fn decode(src: &[u8], dst: &mut [u8], case: Case) -> Result<(), Error> {
    match decode_inner(src, dst, case) {
        0 => Ok(()),
        _ => Err(Error::InvalidEncoding),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::{LOWER, MIXED, UPPER, test_support};

    cpufeatures::new!(detect, "ssse3");

    /// Whether this tier can be exercised on this host.
    ///
    /// If the build claims the feature, this asserts rather than skips: that
    /// combination would otherwise fault at runtime, and a CI leg whose whole
    /// purpose is to force this tier would silently test nothing instead.
    ///
    /// Miri reports no CPU features but does interpret the intrinsics, so a
    /// compile-time `target_feature` is sufficient evidence there.
    fn usable() -> bool {
        if cfg!(target_feature = "ssse3") {
            assert!(
                cfg!(miri) || detect::init().get(),
                "ssse3 enabled at compile time but absent at runtime"
            );
            return true;
        }

        let detected = detect::init().get();
        if !detected {
            std::eprintln!("SKIPPED: ssse3 unavailable on this host");
        }
        detected
    }

    fn run(case: Case) {
        if !usable() {
            return;
        }

        // SAFETY: feature support was just established, and the harness always
        // passes correctly sized buffers.
        let decode_fn = |src: &[u8], dst: &mut [u8]| unsafe { decode(src, dst, case) };
        let encode_fn =
            |src: &[u8], dst: &mut [u8], upper: bool| unsafe { encode(src, dst, upper) };

        test_support::exercise_backend(encode_fn, decode_fn, case);
        test_support::exercise_boundary_sizes(decode_fn, case);

        if case != MIXED {
            test_support::exercise_case_strictness(decode_fn, case);
        }
    }

    #[test]
    fn lower_matches_reference() {
        run(LOWER);
    }

    #[test]
    fn upper_matches_reference() {
        run(UPPER);
    }

    #[test]
    fn mixed_matches_reference() {
        run(MIXED);
    }

    /// Exhaustive table coverage. Skipped under Miri, where 256 x 10
    /// interpreted decodes of a 128-byte buffer is far too slow; real x86 CI
    /// hardware runs it.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn decode_accepts_exactly_its_alphabet() {
        if !usable() {
            return;
        }

        for case in [LOWER, UPPER, MIXED] {
            // SAFETY: feature support was just established.
            test_support::exercise_all_byte_values(
                |src, dst| unsafe { decode(src, dst, case) },
                case,
            );
        }
    }
}
