//! Backend internals exposed for constant-time verification.
//!
//! **Not a stable API.** Exempt from semver; present only when the
//! `ct-internals` feature is enabled, and intended solely for the harness in
//! `ct-tests/`.
//!
//! These are explicit thin wrappers rather than a re-export of the backend
//! modules, so that exactly what the harness needs is exposed and nothing
//! else. Each is `#[inline(always)]`, so the wrapper does not sit between
//! Valgrind and the code under test.
//!
//! What the harness needs is the raw error accumulator, *not* the `Result`.
//! A backend's outer `decode` collapses the accumulator into a `Result` whose
//! discriminant is derived from the input bytes, so with those bytes poisoned
//! memcheck flags the collapse itself and drowns out any real finding. The
//! accumulator can instead be unpoisoned deliberately, declaring "was this
//! valid hex" observable -- which it already is, via the returned `Result`.

use crate::backends;

/// Hex alphabet accepted by a decoder, as a const-generic parameter.
pub type Case = backends::Case;

/// Accept `0-9a-f` only.
pub const LOWER: Case = backends::LOWER;
/// Accept `0-9A-F` only.
pub const UPPER: Case = backends::UPPER;
/// Accept `0-9a-fA-F`.
pub const MIXED: Case = backends::MIXED;

/// Encodes via whichever backend dispatch selects on this host.
#[inline(always)]
pub fn dispatch_encode(src: &[u8], dst: &mut [u8], upper: bool) {
    backends::encode(src, dst, upper);
}

/// Encodes via the portable backend.
#[inline(always)]
pub fn soft_encode(src: &[u8], dst: &mut [u8], upper: bool) {
    backends::soft::encode(src, dst, upper);
}

/// Decodes via the portable backend, returning the raw error accumulator.
#[inline(always)]
pub fn soft_decode_inner<const CASE: Case>(src: &[u8], dst: &mut [u8]) -> u16 {
    backends::soft::decode_inner::<CASE>(src, dst)
}

/// aarch64 NEON entry points.
#[cfg(all(
    target_arch = "aarch64",
    target_feature = "neon",
    not(base16ct_backend = "soft")
))]
pub mod neon {
    use super::Case;
    use crate::backends::aarch64_neon as backend;

    /// Encodes via the NEON backend.
    #[inline(always)]
    pub fn encode(src: &[u8], dst: &mut [u8], upper: bool) {
        backend::encode(src, dst, upper);
    }

    /// Decodes via the NEON backend, returning the raw error accumulator.
    #[inline(always)]
    pub fn decode_inner<const CASE: Case>(src: &[u8], dst: &mut [u8]) -> u8 {
        backend::decode_inner::<CASE>(src, dst)
    }
}

/// WebAssembly `simd128` entry points.
#[cfg(all(
    target_arch = "wasm32",
    target_feature = "simd128",
    not(base16ct_backend = "soft")
))]
pub mod simd128 {
    use super::Case;
    use crate::backends::wasm32_simd128 as backend;

    /// Encodes via the `simd128` backend.
    #[inline(always)]
    pub fn encode(src: &[u8], dst: &mut [u8], upper: bool) {
        backend::encode(src, dst, upper);
    }

    /// Decodes via the `simd128` backend, returning the raw error accumulator.
    #[inline(always)]
    pub fn decode_inner<const CASE: Case>(src: &[u8], dst: &mut [u8]) -> u16 {
        backend::decode_inner::<CASE>(src, dst)
    }
}

/// x86 entry points.
///
/// These take `Case` at runtime, matching the backends themselves: the x86
/// tiers select their lookup tables before the loop, so one compiled copy
/// serves all three alphabets.
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(base16ct_backend = "soft")
))]
// These wrap `unsafe` intrinsics backends; the obligation is forwarded, not
// discharged, so every wrapper is itself `unsafe`.
#[allow(unsafe_code)]
pub mod x86 {
    use super::Case;
    use crate::backends::{x86_avx2, x86_ssse3};

    /// Encodes via the SSSE3 backend.
    ///
    /// # Safety
    ///
    /// The CPU must support SSSE3, and `dst.len()` must be `src.len() * 2`.
    #[inline(always)]
    pub unsafe fn ssse3_encode(src: &[u8], dst: &mut [u8], upper: bool) {
        // SAFETY: forwarded to the caller.
        unsafe { x86_ssse3::encode(src, dst, upper) }
    }

    /// Decodes via the SSSE3 backend, returning the raw error accumulator.
    ///
    /// # Safety
    ///
    /// The CPU must support SSSE3, and `src.len()` must be `dst.len() * 2`.
    #[inline(always)]
    pub unsafe fn ssse3_decode_inner(src: &[u8], dst: &mut [u8], case: Case) -> i32 {
        // SAFETY: forwarded to the caller.
        unsafe { x86_ssse3::decode_inner(src, dst, case) }
    }

    /// Encodes via the AVX2 backend.
    ///
    /// # Safety
    ///
    /// The CPU must support AVX2, and `dst.len()` must be `src.len() * 2`.
    #[inline(always)]
    pub unsafe fn avx2_encode(src: &[u8], dst: &mut [u8], upper: bool) {
        // SAFETY: forwarded to the caller.
        unsafe { x86_avx2::encode(src, dst, upper) }
    }

    /// Decodes via the AVX2 backend, returning the raw error accumulator.
    ///
    /// # Safety
    ///
    /// The CPU must support AVX2, and `src.len()` must be `dst.len() * 2`.
    #[inline(always)]
    pub unsafe fn avx2_decode_inner(src: &[u8], dst: &mut [u8], case: Case) -> i32 {
        // SAFETY: forwarded to the caller.
        unsafe { x86_avx2::decode_inner(src, dst, case) }
    }
}
