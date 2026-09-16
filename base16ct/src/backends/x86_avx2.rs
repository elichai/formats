//! AVX2 backend.
//!
//! Same algorithms as [`super::x86_ssse3`] at 256 bits: 32 input bytes per
//! encode iteration and 32 output bytes per decode iteration. Falls through to
//! the SSSE3 tier for the 16..31 byte middle range and to `soft` below that.
//!
//! AVX2 unpack and pack operate independently within each 128-bit lane, so both
//! directions need a cross-lane fixup that the SSSE3 versions do not.

// This module is intrinsics throughout; `unsafe` is the point of it.
#![allow(unsafe_code)]
#![allow(unsafe_op_in_unsafe_fn)]

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use super::{Case, x86_ssse3};
use crate::Error;

/// Decodes 32 hex characters into 16 bytes, returning `(packed16, check)`.
///
/// The `packuswb` and lane fixup are hoisted to the caller so two chunks can be
/// packed with a single `vpackuswb` using both operands meaningfully, rather
/// than the wasteful self-pairing `pack(p, p)` form.
#[inline(always)]
unsafe fn decode_chunk_256(
    chunk: __m256i,
    delta_check: __m256i,
    delta_rebase: __m256i,
    one: __m256i,
    mask_lo: __m256i,
    weights: __m256i,
) -> (__m256i, __m256i) {
    let vm1 = _mm256_sub_epi8(chunk, one);
    let hash_key = _mm256_and_si256(_mm256_srli_epi16(vm1, 4), mask_lo);

    let check = _mm256_add_epi8(vm1, _mm256_shuffle_epi8(delta_check, hash_key));
    let nibbles = _mm256_add_epi8(vm1, _mm256_shuffle_epi8(delta_rebase, hash_key));

    (_mm256_maddubs_epi16(nibbles, weights), check)
}

/// Encodes `src` into `dst`, 32 input bytes per iteration.
///
/// Carries no `#[target_feature]` so a higher tier can inline it for its tail.
///
/// # Safety
///
/// - The CPU must support AVX2 (the caller must carry
///   `#[target_feature(enable = "avx2")]`).
/// - `dst.len()` must be exactly `src.len() * 2`.
#[inline(always)]
pub(crate) unsafe fn encode_inner(src: &[u8], dst: &mut [u8], upper: bool) {
    debug_assert_eq!(dst.len(), src.len() * 2);

    let lut = _mm256_broadcastsi128_si256(x86_ssse3::hex_lut_128(upper));
    let mask_lo = _mm256_set1_epi8(0x0f);

    let mut done = 0;
    while src.len() - done >= 32 {
        // SAFETY: `done + 32 <= src.len()`, so the 32-byte read is in bounds,
        // and `dst` is twice as long so the two 32-byte writes are too.
        let chunk = _mm256_loadu_si256(src.as_ptr().add(done).cast());

        let lo = _mm256_and_si256(chunk, mask_lo);
        let hi = _mm256_and_si256(_mm256_srli_epi16(chunk, 4), mask_lo);

        let hex_lo = _mm256_shuffle_epi8(lut, lo);
        let hex_hi = _mm256_shuffle_epi8(lut, hi);

        // Correct within each 128-bit lane, but in the wrong lane order.
        let interleaved_lo = _mm256_unpacklo_epi8(hex_hi, hex_lo);
        let interleaved_hi = _mm256_unpackhi_epi8(hex_hi, hex_lo);

        // Reassemble as [lane0_lo, lane0_hi] then [lane1_lo, lane1_hi].
        let out0 = _mm256_permute2x128_si256(interleaved_lo, interleaved_hi, 0x20);
        let out1 = _mm256_permute2x128_si256(interleaved_lo, interleaved_hi, 0x31);

        let out = dst.as_mut_ptr().add(done * 2);
        _mm256_storeu_si256(out.cast(), out0);
        _mm256_storeu_si256(out.add(32).cast(), out1);

        done += 32;
    }

    if done < src.len() {
        x86_ssse3::encode_inner(&src[done..], &mut dst[done * 2..], upper);
    }
}

/// Decodes `src` into `dst`, returning the raw error accumulator.
///
/// # Safety
///
/// - The CPU must support AVX2 (the caller must carry
///   `#[target_feature(enable = "avx2")]`).
/// - `src.len()` must be exactly `dst.len() * 2`.
#[inline(always)]
pub(crate) unsafe fn decode_inner(src: &[u8], dst: &mut [u8], case: Case) -> i32 {
    debug_assert_eq!(src.len(), dst.len() * 2);

    let delta_check = _mm256_broadcastsi128_si256(x86_ssse3::delta_check_128(case));
    let delta_rebase = _mm256_broadcastsi128_si256(x86_ssse3::delta_rebase_128());
    let one = _mm256_set1_epi8(1);
    let mask_lo = _mm256_set1_epi8(0x0f);
    let weights = _mm256_set1_epi16(0x0110);
    let mut err = 0i32;

    // 64 hex characters -> 32 bytes per iteration.
    let mut done = 0;
    while dst.len() - done >= 32 {
        // SAFETY: `done + 32 <= dst.len()` and `src` is twice as long, so both
        // 32-byte reads and the 32-byte write are in bounds.
        let hex = src.as_ptr().add(done * 2);
        let chunk0 = _mm256_loadu_si256(hex.cast());
        let chunk1 = _mm256_loadu_si256(hex.add(32).cast());

        let (packed0, check0) =
            decode_chunk_256(chunk0, delta_check, delta_rebase, one, mask_lo, weights);
        let (packed1, check1) =
            decode_chunk_256(chunk1, delta_check, delta_rebase, one, mask_lo, weights);

        err |= _mm256_movemask_epi8(_mm256_or_si256(check0, check1));

        // `packus(p0, p1)` gives lane 0 = [c0_lo, c1_lo], lane 1 = [c0_hi,
        // c1_hi]; permuting qwords (0, 2, 1, 3) yields c0's 16 bytes followed
        // by c1's.
        let combined = _mm256_packus_epi16(packed0, packed1);
        let result = _mm256_permute4x64_epi64(combined, 0b_11_01_10_00);
        _mm256_storeu_si256(dst.as_mut_ptr().add(done).cast(), result);

        done += 32;
    }

    if done < dst.len() {
        err |= x86_ssse3::decode_inner(&src[done * 2..], &mut dst[done..], case);
    }

    err
}

/// Encodes `src` into `dst` using the upper-case alphabet if `upper` is set.
///
/// # Safety
///
/// - The CPU must support AVX2.
/// - `dst.len()` must be exactly `src.len() * 2`.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn encode(src: &[u8], dst: &mut [u8], upper: bool) {
    encode_inner(src, dst, upper)
}

/// Decodes `src` into `dst`, accepting only the alphabet selected by `case`.
///
/// # Safety
///
/// - The CPU must support AVX2.
/// - `src.len()` must be exactly `dst.len() * 2`.
#[target_feature(enable = "avx2")]
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

    cpufeatures::new!(detect, "avx2");

    /// Whether this tier can be exercised on this host.
    ///
    /// If the build claims the feature, this asserts rather than skips: that
    /// combination would otherwise fault at runtime, and a CI leg whose whole
    /// purpose is to force this tier would silently test nothing instead.
    ///
    /// Miri reports no CPU features but does interpret the intrinsics, so a
    /// compile-time `target_feature` is sufficient evidence there.
    fn usable() -> bool {
        if cfg!(target_feature = "avx2") {
            assert!(
                cfg!(miri) || detect::init().get(),
                "avx2 enabled at compile time but absent at runtime"
            );
            return true;
        }

        let detected = detect::init().get();
        if !detected {
            std::eprintln!("SKIPPED: avx2 unavailable on this host");
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
