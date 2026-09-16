//! Constant-time verification under Valgrind's memcheck.
//!
//! Input buffers are marked `Undefined`, so memcheck reports an error if any
//! branch or address is derived from their contents. The *only* value
//! unpoisoned is the raw error accumulator returned by each backend's
//! `decode_inner`, which declares "was the input valid hex" observable -- it
//! already is, via the returned `Result` -- while leaving the byte values
//! themselves poisoned. That is what makes this a test of the actual claim
//! rather than a test that decoding reports errors.
//!
//! Requires Valgrind, and is meaningless without it: run directly, the
//! poisoning calls are no-ops and everything passes trivially.
//!
//! # What this does and does not prove, per architecture
//!
//! Memcheck's definedness tracking does not reach every register file, and
//! the difference is large enough to state rather than imply. Measured against
//! Valgrind 3.19 by injecting a data-dependent early exit and checking whether
//! it was reported:
//!
//! - **x86 (SSE/AVX2): tracked.** A branch on a value reduced out of a vector
//!   register with `pmovmskb` is reported. The SIMD backends are genuinely
//!   covered here.
//! - **aarch64 (NEON): not tracked.** A branch on a value taken out of a NEON
//!   register -- whether by `vmaxvq_u8` or a plain per-lane `vgetq_lane_u8` --
//!   is *not* reported, so definedness is lost as soon as data enters a `Q`
//!   register. A scalar branch on the same poisoned input inside the same
//!   function *is* reported, so the poisoning itself is working.
//!
//! So on `aarch64` this covers the portable backend fully, plus anything the
//! NEON backend does outside the vector registers (loop bounds, tail
//! dispatch, scalar branches). It does not cover a branch on a value computed
//! through NEON lanes. The job is kept because those are real properties worth
//! regression-testing, not because it is equivalent to the x86 job.
//!
//! The NEON backend's constant-time argument therefore rests on the code
//! being branch-free by construction, reviewed as such, and on the x86
//! backends -- which implement the same algorithms -- being checked here.
//!
//! # Public API coverage
//!
//! Encode is driven through the real public API, because it has no
//! data-dependent error path: `Error::InvalidLength` depends only on lengths.
//! Nothing has to be declassified, and the wrappers, `from_utf8_unchecked`,
//! the allocating paths and `HexDisplay` are all covered.
//!
//! Decode is not, and cannot be as things stand. Collapsing the accumulator
//! into a `Result` produces a discriminant derived from the input bytes, which
//! memcheck reports whether it compiles to a branch or to a conditional move.
//! That report is a false positive against this crate's documented scope --
//! whether the input was valid hex is already public, via the returned
//! `Result` -- but suppressing it from outside the library is not possible,
//! because the value lives in a register with no address to mark.
//!
//! Making the public decode API testable would need *declassification*: the
//! library marking the accumulator defined immediately before the comparison,
//! the way BoringSSL's `CONSTTIME_DECLASSIFY` does. That is a standard
//! technique, and it would extend this coverage to decode's wrappers -- which
//! matters most on `aarch64`, where those wrappers are scalar and therefore
//! visible to memcheck even though the SIMD lanes are not. It is not done here
//! because it requires instrumentation inside `base16ct`, and a Valgrind
//! dependency behind a `cfg`, in a crate that otherwise has neither.
//!
//! ```sh
//! cd base16ct/ct-tests
//! CARGO_TARGET_$(rustc -vV | sed -n 's/host: //p' | tr 'a-z-' 'A-Z_')_RUNNER=\
//!   "valgrind --tool=memcheck --error-exitcode=1" cargo test --release
//! ```

use base16ct::ct_internals as ct;
use crabgrind::memcheck;
use std::ffi::c_void;
use std::fmt::Debug;

/// Largest input exercised.
///
/// Smaller than the correctness harness on purpose: a data-dependent branch
/// shows up at essentially any size past one chunk, so the sweep only needs to
/// cover every chunk width and a few iterations of the widest loop. Paying
/// Valgrind's slowdown for a 512-byte sweep buys nothing.
const MAX_SIZE: usize = 128;

/// Sizes straddling each SIMD chunk width, for the quadratic checks.
const BOUNDARY_SIZES: [usize; 12] = [1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128];

/// A byte that is not a hex digit in any case.
const INVALID: u8 = 0xff;

fn fill_pseudorandom(buf: &mut [u8], seed: u64) {
    let mut s = seed | 1;
    for b in buf.iter_mut() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        *b = (s >> 33) as u8;
    }
}

/// Marks `data` as undefined, so any use of its contents is an error.
fn poison(data: &[u8]) {
    memcheck::mark_memory(
        data.as_ptr().cast::<c_void>(),
        data.len(),
        memcheck::MemState::Undefined,
    )
    .unwrap_or(());
}

/// Declares `val` observable again, undoing [`poison`] for that value alone.
fn unpoison<T: ?Sized>(val: &mut T) {
    let len = size_of_val(val);
    let ptr = (val as *mut T).cast::<c_void>();
    memcheck::mark_memory(ptr, len, memcheck::MemState::Defined).unwrap_or(());
}

/// Encodes `src` in whichever alphabet `case` accepts, without poisoning.
fn reference_hex(src: &[u8], dst: &mut [u8], case: ct::Case) {
    ct::soft_encode(src, dst, case == ct::UPPER);

    if case == ct::MIXED {
        for (i, b) in dst.iter_mut().enumerate() {
            if i % 2 == 0 {
                b.make_ascii_uppercase();
            }
        }
    }
}

/// Encodes poisoned input at every size; memcheck flags any branch on content.
fn encode_is_constant_time(encode: impl Fn(&[u8], &mut [u8], bool)) {
    let mut input = [0u8; MAX_SIZE];
    let mut output = [0u8; MAX_SIZE * 2];

    for size in 0..=MAX_SIZE {
        let input = &mut input[..size];
        fill_pseudorandom(input, size as u64 + 1);
        let output = &mut output[..size * 2];

        for upper in [false, true] {
            poison(input);
            poison(output);
            encode(input, output, upper);
            unpoison(output);
        }
    }
}

/// Decodes poisoned *valid* input, unpoisoning only the accumulator.
fn decode_is_constant_time<T>(decode: impl Fn(&[u8], &mut [u8]) -> T, case: ct::Case)
where
    T: PartialEq + Default + Copy + Debug,
{
    let mut input = [0u8; MAX_SIZE];
    let mut hex = [0u8; MAX_SIZE * 2];
    let mut output = [0u8; MAX_SIZE];

    for size in 0..=MAX_SIZE {
        let input = &mut input[..size];
        fill_pseudorandom(input, size as u64 + 1);

        let hex = &mut hex[..size * 2];
        reference_hex(input, hex, case);
        let output = &mut output[..size];

        poison(hex);
        poison(output);
        let mut accum = decode(hex, output);
        unpoison(&mut accum);
        unpoison(output);

        assert_eq!(
            accum,
            T::default(),
            "valid input of size {size} reported an error, case {case}"
        );
    }
}

/// Decodes poisoned input with an invalid byte at every position.
///
/// This is what catches an early exit: the loop bound would then depend on
/// poisoned content rather than on the length, which memcheck reports.
fn decode_invalid_is_constant_time<T>(decode: impl Fn(&[u8], &mut [u8]) -> T, case: ct::Case)
where
    T: PartialEq + Default + Copy + Debug,
{
    let mut input = [0u8; MAX_SIZE];
    let mut hex = [0u8; MAX_SIZE * 2];
    let mut pristine = [0u8; MAX_SIZE * 2];
    let mut output = [0u8; MAX_SIZE];

    for size in BOUNDARY_SIZES {
        let input = &mut input[..size];
        fill_pseudorandom(input, size as u64 + 1);

        let hex = &mut hex[..size * 2];
        reference_hex(input, hex, case);
        pristine[..size * 2].copy_from_slice(hex);
        let output = &mut output[..size];

        for pos in 0..size * 2 {
            hex.copy_from_slice(&pristine[..size * 2]);
            hex[pos] = INVALID;

            poison(hex);
            poison(output);
            let mut accum = decode(hex, output);
            unpoison(&mut accum);
            unpoison(output);

            assert_ne!(
                accum,
                T::default(),
                "invalid byte at {pos} of size {size} went unreported, case {case}"
            );
        }
    }
}

/// Exercises the *public* encode API, end to end.
///
/// Encode has no data-dependent error path -- `Error::InvalidLength` depends
/// only on lengths -- so no value has to be declassified and the real API can
/// be driven directly. This covers the wrappers the per-backend tests skip:
/// the length checks, `from_utf8_unchecked`, the allocating paths, and
/// `HexDisplay`.
#[test]
fn public_encode_api_is_constant_time() {
    let mut input = [0u8; MAX_SIZE];
    let mut output = [0u8; MAX_SIZE * 2];

    for size in 0..=MAX_SIZE {
        let input = &mut input[..size];
        fill_pseudorandom(input, size as u64 + 1);
        let output = &mut output[..size * 2];

        poison(input);

        base16ct::lower::encode(input, output).expect("correctly sized");
        base16ct::upper::encode(input, output).expect("correctly sized");
        base16ct::lower::encode_str(input, output).expect("correctly sized");
        base16ct::upper::encode_str(input, output).expect("correctly sized");

        let mut lower = base16ct::lower::encode_string(input);
        let mut upper = base16ct::upper::encode_string(input);
        unpoison(lower.as_mut_str());
        unpoison(upper.as_mut_str());

        unpoison(output);
        unpoison(input);
    }
}

/// `HexDisplay` goes through `core::fmt`, so it is checked separately: a
/// finding here could be in the formatter rather than in this crate.
#[test]
fn hex_display_is_constant_time() {
    use std::fmt::Write as _;

    let mut input = [0u8; MAX_SIZE];
    let mut out = String::with_capacity(MAX_SIZE * 2);

    for size in 0..=MAX_SIZE {
        let input = &mut input[..size];
        fill_pseudorandom(input, size as u64 + 1);

        poison(input);
        out.clear();
        write!(out, "{:x}", base16ct::HexDisplay(input)).expect("infallible");
        write!(out, "{:X}", base16ct::HexDisplay(input)).expect("infallible");
        unpoison(out.as_mut_str());
        unpoison(input);
    }
}

#[test]
fn soft_is_constant_time() {
    encode_is_constant_time(ct::soft_encode);

    decode_is_constant_time(ct::soft_decode_inner::<{ ct::LOWER }>, ct::LOWER);
    decode_is_constant_time(ct::soft_decode_inner::<{ ct::UPPER }>, ct::UPPER);
    decode_is_constant_time(ct::soft_decode_inner::<{ ct::MIXED }>, ct::MIXED);

    decode_invalid_is_constant_time(ct::soft_decode_inner::<{ ct::LOWER }>, ct::LOWER);
    decode_invalid_is_constant_time(ct::soft_decode_inner::<{ ct::MIXED }>, ct::MIXED);
}

/// Encode through whichever backend dispatch selects on this host.
///
/// Decode is not covered here: the dispatch layer returns a `Result`, whose
/// discriminant is derived from poisoned data, so memcheck would flag the
/// collapse rather than anything interesting. The per-backend tests below
/// observe the accumulators directly instead.
#[test]
fn dispatch_encode_is_constant_time() {
    encode_is_constant_time(ct::dispatch_encode);
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[test]
fn neon_is_constant_time() {
    use base16ct::ct_internals::neon;

    encode_is_constant_time(neon::encode);

    decode_is_constant_time(neon::decode_inner::<{ ct::LOWER }>, ct::LOWER);
    decode_is_constant_time(neon::decode_inner::<{ ct::UPPER }>, ct::UPPER);
    decode_is_constant_time(neon::decode_inner::<{ ct::MIXED }>, ct::MIXED);

    decode_invalid_is_constant_time(neon::decode_inner::<{ ct::LOWER }>, ct::LOWER);
    decode_invalid_is_constant_time(neon::decode_inner::<{ ct::MIXED }>, ct::MIXED);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    use super::*;
    use base16ct::ct_internals::x86;

    cpufeatures::new!(has_ssse3, "ssse3");
    cpufeatures::new!(has_avx2, "avx2");
    cpufeatures::new!(has_avx512bw, "avx512bw");
    cpufeatures::new!(has_avx512vbmi, "avx512bw", "avx512vbmi");

    #[test]
    fn ssse3_is_constant_time() {
        if !(cfg!(target_feature = "ssse3") || has_ssse3::init().get()) {
            eprintln!("SKIPPED: ssse3 unavailable on this host");
            return;
        }

        // SAFETY: SSSE3 availability was just established, and the drivers
        // always pass correctly sized buffers.
        encode_is_constant_time(|src, dst, upper| unsafe { x86::ssse3_encode(src, dst, upper) });

        for case in [ct::LOWER, ct::UPPER, ct::MIXED] {
            decode_is_constant_time(
                |src, dst| unsafe { x86::ssse3_decode_inner(src, dst, case) },
                case,
            );
        }

        for case in [ct::LOWER, ct::MIXED] {
            decode_invalid_is_constant_time(
                |src, dst| unsafe { x86::ssse3_decode_inner(src, dst, case) },
                case,
            );
        }
    }

    #[test]
    fn avx512_is_constant_time() {
        // Decode needs only AVX-512BW; encode also needs VBMI.
        if !(cfg!(target_feature = "avx512bw") || has_avx512bw::init().get()) {
            eprintln!("SKIPPED: avx512bw unavailable on this host");
            return;
        }

        if cfg!(all(
            target_feature = "avx512bw",
            target_feature = "avx512vbmi"
        )) || has_avx512vbmi::init().get()
        {
            // SAFETY: AVX-512BW and VBMI were just established.
            encode_is_constant_time(|src, dst, upper| unsafe {
                x86::avx512_encode(src, dst, upper)
            });
        } else {
            eprintln!("SKIPPED (encode only): avx512vbmi unavailable on this host");
        }

        // SAFETY: AVX-512BW was just established, and the drivers always pass
        // correctly sized buffers.
        for case in [ct::LOWER, ct::UPPER, ct::MIXED] {
            decode_is_constant_time(
                |src, dst| unsafe { x86::avx512_decode_inner(src, dst, case) },
                case,
            );
        }

        for case in [ct::LOWER, ct::MIXED] {
            decode_invalid_is_constant_time(
                |src, dst| unsafe { x86::avx512_decode_inner(src, dst, case) },
                case,
            );
        }
    }

    #[test]
    fn avx2_is_constant_time() {
        if !(cfg!(target_feature = "avx2") || has_avx2::init().get()) {
            eprintln!("SKIPPED: avx2 unavailable on this host");
            return;
        }

        // SAFETY: AVX2 availability was just established, and the drivers
        // always pass correctly sized buffers.
        encode_is_constant_time(|src, dst, upper| unsafe { x86::avx2_encode(src, dst, upper) });

        for case in [ct::LOWER, ct::UPPER, ct::MIXED] {
            decode_is_constant_time(
                |src, dst| unsafe { x86::avx2_decode_inner(src, dst, case) },
                case,
            );
        }

        for case in [ct::LOWER, ct::MIXED] {
            decode_invalid_is_constant_time(
                |src, dst| unsafe { x86::avx2_decode_inner(src, dst, case) },
                case,
            );
        }
    }
}
