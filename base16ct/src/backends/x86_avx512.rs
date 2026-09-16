//! AVX-512 backend.
//!
//! Same algorithms as [`super::x86_ssse3`] at 512 bits, falling through to the
//! AVX2 tier for the 32..63 byte middle range and downwards from there.
//!
//! The two directions need different feature levels, which is why they are
//! detected separately by the dispatch layer:
//!
//! - **decode** needs only AVX-512BW. `vpmovwb` narrows 32 `i16` lanes
//!   straight into a `__m256i`, replacing the `packuswb` plus cross-lane
//!   `vpermq` fixup the AVX2 path needs, and `vpmovb2m` collects the error
//!   mask into a `u64` with no horizontal reduction at all.
//! - **encode** additionally needs AVX-512VBMI, for `vpermi2b`. That does the
//!   whole high-nibble/low-nibble interleave as a single byte-granular
//!   cross-lane permute per output register.

// This module is intrinsics throughout; `unsafe` is the point of it.
#![allow(unsafe_code)]
#![allow(unsafe_op_in_unsafe_fn)]

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use super::{Case, x86_avx2, x86_ssse3};
use crate::Error;

/// `vpermi2b` indices placing output bytes `0..64`.
///
/// Input byte `k` contributes its high-nibble character at output `2k` (index
/// `k`, first source) and its low-nibble character at `2k + 1` (index
/// `64 + k`, second source).
const INTERLEAVE_LO: [i8; 64] = {
    let mut a = [0i8; 64];
    let mut i = 0;
    while i < 32 {
        a[i * 2] = i as i8;
        a[i * 2 + 1] = (64 + i) as i8;
        i += 1;
    }
    a
};

/// `vpermi2b` indices placing output bytes `64..128`, from input bytes `32..64`.
const INTERLEAVE_HI: [i8; 64] = {
    let mut a = [0i8; 64];
    let mut i = 0;
    while i < 32 {
        a[i * 2] = (32 + i) as i8;
        a[i * 2 + 1] = (96 + i) as i8;
        i += 1;
    }
    a
};

/// Decodes 64 hex characters into 32 bytes, returning the bytes and the
/// one-bit-per-lane error mask.
#[inline(always)]
unsafe fn decode_chunk_512(
    chunk: __m512i,
    delta_check: __m512i,
    delta_rebase: __m512i,
    one: __m512i,
    mask_lo: __m512i,
    weights: __m512i,
) -> (__m256i, u64) {
    let vm1 = _mm512_sub_epi8(chunk, one);
    let hash_key = _mm512_and_si512(_mm512_srli_epi16(vm1, 4), mask_lo);

    let check = _mm512_add_epi8(vm1, _mm512_shuffle_epi8(delta_check, hash_key));
    let nibbles = _mm512_add_epi8(vm1, _mm512_shuffle_epi8(delta_rebase, hash_key));

    // One bit per byte, set where the MSB was set, i.e. where input was invalid.
    let mask = _mm512_movepi8_mask(check);

    // hi * 16 + lo, then narrow 32 i16 lanes directly to u8 with no fixup.
    let packed16 = _mm512_maddubs_epi16(nibbles, weights);

    (_mm512_cvtepi16_epi8(packed16), mask)
}

/// Encodes `src` into `dst`, 64 input bytes per iteration.
///
/// Carries no `#[target_feature]` so a higher tier could inline it for its
/// tail.
///
/// # Safety
///
/// - The CPU must support AVX-512BW and AVX-512VBMI (the caller must carry
///   `#[target_feature(enable = "avx512bw,avx512vbmi")]`).
/// - `dst.len()` must be exactly `src.len() * 2`.
#[inline(always)]
pub(crate) unsafe fn encode_inner(src: &[u8], dst: &mut [u8], upper: bool) {
    debug_assert_eq!(dst.len(), src.len() * 2);

    let lut = _mm512_broadcast_i32x4(x86_ssse3::hex_lut_128(upper));
    let mask_lo = _mm512_set1_epi8(0x0f);
    let idx0 = _mm512_loadu_si512(INTERLEAVE_LO.as_ptr().cast());
    let idx1 = _mm512_loadu_si512(INTERLEAVE_HI.as_ptr().cast());

    let mut done = 0;
    while src.len() - done >= 64 {
        // SAFETY: `done + 64 <= src.len()`, so the 64-byte read is in bounds,
        // and `dst` is twice as long so the two 64-byte writes are too.
        let chunk = _mm512_loadu_si512(src.as_ptr().add(done).cast());

        let lo = _mm512_and_si512(chunk, mask_lo);
        let hi = _mm512_and_si512(_mm512_srli_epi16(chunk, 4), mask_lo);

        let hex_lo = _mm512_shuffle_epi8(lut, lo);
        let hex_hi = _mm512_shuffle_epi8(lut, hi);

        // One byte-granular cross-lane permute per output register.
        let out0 = _mm512_permutex2var_epi8(hex_hi, idx0, hex_lo);
        let out1 = _mm512_permutex2var_epi8(hex_hi, idx1, hex_lo);

        let out = dst.as_mut_ptr().add(done * 2);
        _mm512_storeu_si512(out.cast(), out0);
        _mm512_storeu_si512(out.add(64).cast(), out1);

        done += 64;
    }

    if done < src.len() {
        x86_avx2::encode_inner(&src[done..], &mut dst[done * 2..], upper);
    }
}

/// Decodes `src` into `dst`, returning the raw error accumulator.
///
/// # Safety
///
/// - The CPU must support AVX-512BW (the caller must carry
///   `#[target_feature(enable = "avx512bw")]`).
/// - `src.len()` must be exactly `dst.len() * 2`.
#[inline(always)]
pub(crate) unsafe fn decode_inner(src: &[u8], dst: &mut [u8], case: Case) -> u64 {
    debug_assert_eq!(src.len(), dst.len() * 2);

    let delta_check = _mm512_broadcast_i32x4(x86_ssse3::delta_check_128(case));
    let delta_rebase = _mm512_broadcast_i32x4(x86_ssse3::delta_rebase_128());
    let one = _mm512_set1_epi8(1);
    let mask_lo = _mm512_set1_epi8(0x0f);
    let weights = _mm512_set1_epi16(0x0110);
    let mut err = 0u64;

    // 128 hex characters -> 64 bytes per iteration.
    let mut done = 0;
    while dst.len() - done >= 64 {
        // SAFETY: `done + 64 <= dst.len()` and `src` is twice as long, so both
        // 64-byte reads and both 32-byte writes are in bounds.
        let hex = src.as_ptr().add(done * 2);
        let chunk0 = _mm512_loadu_si512(hex.cast());
        let chunk1 = _mm512_loadu_si512(hex.add(64).cast());

        let (decoded0, mask0) =
            decode_chunk_512(chunk0, delta_check, delta_rebase, one, mask_lo, weights);
        let (decoded1, mask1) =
            decode_chunk_512(chunk1, delta_check, delta_rebase, one, mask_lo, weights);

        err |= mask0 | mask1;

        let out = dst.as_mut_ptr().add(done);
        _mm256_storeu_si256(out.cast(), decoded0);
        _mm256_storeu_si256(out.add(32).cast(), decoded1);

        done += 64;
    }

    if done < dst.len() {
        // Widen through `u32`: the AVX2 accumulator is an `i32` whose top bit
        // can be set, and a direct `as u64` would sign-extend. Either way a
        // nonzero stays nonzero, but not by accident.
        let tail = x86_avx2::decode_inner(&src[done * 2..], &mut dst[done..], case);
        err |= u64::from(tail as u32);
    }

    err
}

/// Encodes `src` into `dst` using the upper-case alphabet if `upper` is set.
///
/// # Safety
///
/// - The CPU must support AVX-512BW and AVX-512VBMI.
/// - `dst.len()` must be exactly `src.len() * 2`.
#[target_feature(enable = "avx512bw,avx512vbmi")]
pub(crate) unsafe fn encode(src: &[u8], dst: &mut [u8], upper: bool) {
    encode_inner(src, dst, upper)
}

/// Decodes `src` into `dst`, accepting only the alphabet selected by `case`.
///
/// # Safety
///
/// - The CPU must support AVX-512BW.
/// - `src.len()` must be exactly `dst.len() * 2`.
#[target_feature(enable = "avx512bw")]
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

    cpufeatures::new!(has_bw, "avx512bw");
    cpufeatures::new!(has_vbmi, "avx512bw", "avx512vbmi");

    /// Decode needs only AVX-512BW; see the module documentation.
    fn decode_usable() -> bool {
        if cfg!(target_feature = "avx512bw") {
            assert!(
                cfg!(miri) || has_bw::init().get(),
                "avx512bw enabled at compile time but absent at runtime"
            );
            return true;
        }

        let detected = has_bw::init().get();
        if !detected {
            std::eprintln!("SKIPPED: avx512bw unavailable on this host");
        }
        detected
    }

    /// Encode additionally needs AVX-512VBMI, for `vpermi2b`.
    fn encode_usable() -> bool {
        if cfg!(all(
            target_feature = "avx512bw",
            target_feature = "avx512vbmi"
        )) {
            assert!(
                cfg!(miri) || has_vbmi::init().get(),
                "avx512vbmi enabled at compile time but absent at runtime"
            );
            return true;
        }

        let detected = has_vbmi::init().get();
        if !detected {
            std::eprintln!("SKIPPED: avx512vbmi unavailable on this host");
        }
        detected
    }

    fn run(case: Case) {
        if !decode_usable() {
            return;
        }

        // SAFETY: feature support was just established, and the harness always
        // passes correctly sized buffers.
        let decode_fn = |src: &[u8], dst: &mut [u8]| unsafe { decode(src, dst, case) };

        if encode_usable() {
            let encode_fn =
                |src: &[u8], dst: &mut [u8], upper: bool| unsafe { encode(src, dst, upper) };
            test_support::exercise_backend(encode_fn, decode_fn, case);
        }

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

    /// Exhaustive table coverage. Skipped under Miri, where interpreting it is
    /// far too slow; real AVX-512 hardware or Intel SDE runs it.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn decode_accepts_exactly_its_alphabet() {
        if !decode_usable() {
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
