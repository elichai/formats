//! Shared harness for exercising a backend against a naive reference.
//!
//! Every backend is driven through the same checks, so a SIMD backend only has
//! to supply closures pointing at its own entry points.

#![deny(unsafe_code)]

use super::{Case, LOWER, MIXED, UPPER};
use crate::Error;

/// Largest input exercised. Miri interprets every operation, so stay small there.
pub(crate) const MAX_SIZE: usize = if cfg!(miri) { 64 } else { 512 };

/// Sizes straddling every SIMD chunk width, plus one byte either side.
pub(crate) const BOUNDARY_SIZES: [usize; 12] = [15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129];

/// A byte that is not a hex digit in any case.
pub(crate) const INVALID: u8 = b'G';

/// Written into output buffers before each decode.
///
/// Without this, a decoder that returns early leaves correct values behind from
/// a previous decode, and the no-early-exit check silently passes.
pub(crate) const POISON: u8 = 0xAA;

/// Fills `buf` with a reproducible pseudorandom pattern (xorshift64).
///
/// Hand-rolled to keep the crate free of a PRNG dev-dependency.
pub(crate) fn fill_pseudorandom(buf: &mut [u8], seed: u64) {
    let mut s = seed | 1;
    for b in buf.iter_mut() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        *b = (s >> 33) as u8;
    }
}

/// Straightforward table-driven encoder, used only as a test oracle.
pub(crate) fn naive_encode(src: &[u8], dst: &mut [u8], upper: bool) {
    let lut: &[u8; 16] = if upper {
        b"0123456789ABCDEF"
    } else {
        b"0123456789abcdef"
    };

    for (s, d) in src.iter().zip(dst.chunks_exact_mut(2)) {
        d[0] = lut[usize::from(s >> 4)];
        d[1] = lut[usize::from(s & 0x0f)];
    }
}

/// Straightforward decoder, used only as a test oracle.
pub(crate) fn naive_decode_nibble(byte: u8, case: Case) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' if case != UPPER => Some(byte - b'a' + 10),
        b'A'..=b'F' if case != LOWER => Some(byte - b'A' + 10),
        _ => None,
    }
}

/// Decodes `src` into `dst`, returning whether every byte was in the alphabet.
pub(crate) fn naive_decode(src: &[u8], dst: &mut [u8], case: Case) -> bool {
    let mut valid = true;

    for (s, d) in src.chunks_exact(2).zip(dst.iter_mut()) {
        match (
            naive_decode_nibble(s[0], case),
            naive_decode_nibble(s[1], case),
        ) {
            (Some(hi), Some(lo)) => *d = (hi << 4) | lo,
            _ => valid = false,
        }
    }

    valid
}

/// Encodes `src` in whichever alphabet `case` accepts.
///
/// `MIXED` alternates case per nibble, so a mixed-case decoder is actually
/// exercised on mixed input rather than on uniformly lower-case input.
fn encode_for_case(src: &[u8], dst: &mut [u8], case: Case) {
    naive_encode(src, dst, case == UPPER);

    if case == MIXED {
        for (i, b) in dst.iter_mut().enumerate() {
            if i % 2 == 0 {
                b.make_ascii_uppercase();
            }
        }
    }
}

/// Compares a backend against the naive reference across `0..=MAX_SIZE`.
///
/// `encode` and `decode` must write exactly `src.len() * 2` and `src.len() / 2`
/// bytes respectively; `decode` is the backend's entry point for `case`.
pub(crate) fn exercise_backend<E, D>(encode: E, decode: D, case: Case)
where
    E: Fn(&[u8], &mut [u8], bool),
    D: Fn(&[u8], &mut [u8]) -> Result<(), Error>,
{
    let mut input = [0u8; MAX_SIZE];
    let mut hex = [0u8; MAX_SIZE * 2];
    let mut reference_hex = [0u8; MAX_SIZE * 2];
    let mut decoded = [0u8; MAX_SIZE];
    let mut reference = [0u8; MAX_SIZE];

    for size in 0..=MAX_SIZE {
        let hex_len = size * 2;
        let input = &mut input[..size];
        fill_pseudorandom(input, size as u64 + 1);

        // Encode matches the reference in both alphabets.
        for upper in [false, true] {
            let hex = &mut hex[..hex_len];
            let reference_hex = &mut reference_hex[..hex_len];

            encode(input, hex, upper);
            naive_encode(input, reference_hex, upper);
            assert_eq!(hex, reference_hex, "encode mismatch at size {size}");
        }

        // Valid input round-trips.
        let hex = &mut hex[..hex_len];
        encode_for_case(input, hex, case);

        let decoded = &mut decoded[..size];
        decoded.fill(POISON);
        decode(hex, decoded).expect("valid input rejected");
        assert_eq!(decoded, input, "decode mismatch at size {size}");

        // A single invalid byte is rejected wherever it appears, and every
        // unaffected output byte still decodes correctly. Both properties
        // together are what "no early exit" actually means.
        if hex_len == 0 {
            continue;
        }

        let reference = &mut reference[..size];

        for pos in [0, hex_len / 2, hex_len - 1] {
            let saved = hex[pos];
            hex[pos] = INVALID;

            assert!(!naive_decode(hex, reference, case));
            decoded.fill(POISON);
            assert_eq!(
                decode(hex, decoded),
                Err(Error::InvalidEncoding),
                "invalid byte at {pos} of {hex_len} accepted"
            );

            for (i, (got, want)) in decoded.iter().zip(reference.iter()).enumerate() {
                if i != pos / 2 {
                    assert_eq!(got, want, "byte {i} clobbered by invalid input at {pos}");
                }
            }

            hex[pos] = saved;
        }
    }
}

/// Asserts a strict decoder rejects the alphabet it does not accept.
///
/// Only meaningful where the two encodings actually differ: a value made
/// entirely of `0-9` nibbles encodes identically in both cases.
pub(crate) fn exercise_case_strictness<D>(decode: D, case: Case)
where
    D: Fn(&[u8], &mut [u8]) -> Result<(), Error>,
{
    assert_ne!(case, MIXED, "mixed accepts both alphabets by definition");

    let mut input = [0u8; MAX_SIZE];
    let mut wrong = [0u8; MAX_SIZE * 2];
    let mut decoded = [0u8; MAX_SIZE];
    let mut checked = 0usize;

    for size in 1..=MAX_SIZE {
        let input = &mut input[..size];
        fill_pseudorandom(input, size as u64 + 1);

        let wrong = &mut wrong[..size * 2];
        naive_encode(input, wrong, case != UPPER);

        // Skip sizes whose encoding has no alphabetic nibble.
        if !wrong.iter().any(u8::is_ascii_alphabetic) {
            continue;
        }

        assert_eq!(
            decode(wrong, &mut decoded[..size]),
            Err(Error::InvalidEncoding),
            "wrong-case input accepted at size {size}"
        );
        checked += 1;
    }

    // Every size above skips when its encoding happens to contain no
    // alphabetic nibble, so without this the whole check could pass having
    // asserted nothing.
    assert!(checked > 0, "case strictness was never actually exercised");
}

/// Runs `exercise_backend` over each boundary size with an invalid byte at
/// *every* position, rather than the three sampled by `exercise_backend`.
pub(crate) fn exercise_boundary_sizes<D>(decode: D, case: Case)
where
    D: Fn(&[u8], &mut [u8]) -> Result<(), Error>,
{
    let mut input = [0u8; MAX_SIZE];
    let mut hex = [0u8; MAX_SIZE * 2];
    let mut decoded = [0u8; MAX_SIZE];
    let mut checked = 0usize;

    for size in BOUNDARY_SIZES {
        // Buffers are sized for MAX_SIZE, which Miri lowers; skip the boundaries
        // that do not fit there.
        if size > MAX_SIZE {
            continue;
        }

        let input = &mut input[..size];
        fill_pseudorandom(input, size as u64 + 1);

        let hex = &mut hex[..size * 2];
        let decoded = &mut decoded[..size];
        encode_for_case(input, hex, case);
        decoded.fill(POISON);

        decode(hex, decoded).expect("valid input rejected");
        assert_eq!(decoded, input, "decode mismatch at boundary size {size}");

        for pos in 0..hex.len() {
            let saved = hex[pos];
            hex[pos] = INVALID;

            assert_eq!(
                decode(hex, decoded),
                Err(Error::InvalidEncoding),
                "invalid byte at {pos} of {} accepted",
                hex.len()
            );

            hex[pos] = saved;
        }

        checked += 1;
    }

    // Guards against a lowered `MAX_SIZE` skipping every boundary.
    assert!(checked > 0, "no boundary size was actually exercised");
}

/// Feeds every one of the 256 byte values through a buffer large enough to
/// engage every SIMD tier, at positions spanning each chunk boundary.
///
/// [`exercise_backend`] only ever injects [`INVALID`], which hashes into one
/// table bucket. The SIMD decoders are table-driven and have entries for
/// buckets that no valid character reaches, so a wrong entry in one of those
/// would otherwise go unnoticed: `0x2f`/`0x3a` straddle the digits, `0x40`/
/// `0x47` and `0x60`/`0x67` straddle the two alphabetic ranges, and everything
/// from `0x80` up lands in the high buckets.
pub(crate) fn exercise_all_byte_values<D>(decode: D, case: Case)
where
    D: Fn(&[u8], &mut [u8]) -> Result<(), Error>,
{
    /// 128 hex characters, enough for the widest tier plus a tail.
    const BYTES: usize = 64;

    let mut input = [0u8; BYTES];
    let mut hex = [0u8; BYTES * 2];
    let mut decoded = [0u8; BYTES];
    let mut reference = [0u8; BYTES];

    fill_pseudorandom(&mut input, 7);
    encode_for_case(&input, &mut hex, case);
    let pristine = hex;

    // Every position, not a sample: which SIMD lane a byte lands in affects
    // how it is processed, so a lane-specific fault would survive sampling.
    for value in 0..=u8::MAX {
        for pos in 0..hex.len() {
            hex = pristine;
            hex[pos] = value;

            let want = naive_decode(&hex, &mut reference, case);
            decoded.fill(POISON);
            let got = decode(&hex, &mut decoded);

            assert_eq!(
                got.is_ok(),
                want,
                "byte {value:#04x} at position {pos}, case {case}"
            );

            if want {
                assert_eq!(
                    decoded, reference,
                    "byte {value:#04x} at position {pos}, case {case}"
                );
            }
        }
    }
}
