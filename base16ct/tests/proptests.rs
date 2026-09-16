//! `proptest`-powered property-based tests.
//!
//! Compared against a naive reference implemented here rather than against an
//! external crate, so that `base16ct` keeps its short dependency list.

#![cfg(feature = "alloc")]

use base16ct::Error;
use proptest::{collection::vec, prelude::*};

/// Straightforward table-driven encoder, used only as an oracle.
fn naive_encode(src: &[u8], upper: bool) -> String {
    let lut: &[u8; 16] = if upper {
        b"0123456789ABCDEF"
    } else {
        b"0123456789abcdef"
    };

    let mut out = String::with_capacity(src.len() * 2);
    for b in src {
        out.push(lut[usize::from(b >> 4)] as char);
        out.push(lut[usize::from(b & 0x0f)] as char);
    }
    out
}

/// Which alphabet a decoder accepts.
#[derive(Copy, Clone, PartialEq)]
enum Case {
    Lower,
    Upper,
    Mixed,
}

fn naive_decode_nibble(byte: u8, case: Case) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' if case != Case::Upper => Some(byte - b'a' + 10),
        b'A'..=b'F' if case != Case::Lower => Some(byte - b'A' + 10),
        _ => None,
    }
}

/// Straightforward decoder, used only as an oracle.
fn naive_decode(src: &[u8], case: Case) -> Option<Vec<u8>> {
    if !src.len().is_multiple_of(2) {
        return None;
    }

    src.chunks_exact(2)
        .map(|pair| {
            let hi = naive_decode_nibble(pair[0], case)?;
            let lo = naive_decode_nibble(pair[1], case)?;
            Some((hi << 4) | lo)
        })
        .collect()
}

proptest! {
    #[test]
    fn lower_roundtrip(input in vec(any::<u8>(), 0..256)) {
        let hex = base16ct::lower::encode_string(&input);
        prop_assert_eq!(base16ct::lower::decode_vec(&hex).unwrap(), input);
    }

    #[test]
    fn upper_roundtrip(input in vec(any::<u8>(), 0..256)) {
        let hex = base16ct::upper::encode_string(&input);
        prop_assert_eq!(base16ct::upper::decode_vec(&hex).unwrap(), input);
    }

    #[test]
    fn mixed_accepts_either_alphabet(input in vec(any::<u8>(), 0..256)) {
        let lower = base16ct::lower::encode_string(&input);
        let upper = base16ct::upper::encode_string(&input);
        prop_assert_eq!(base16ct::mixed::decode_vec(&lower).unwrap(), input.clone());
        prop_assert_eq!(base16ct::mixed::decode_vec(&upper).unwrap(), input);
    }

    #[test]
    fn encode_matches_naive(input in vec(any::<u8>(), 0..256)) {
        prop_assert_eq!(base16ct::lower::encode_string(&input), naive_encode(&input, false));
        prop_assert_eq!(base16ct::upper::encode_string(&input), naive_encode(&input, true));
    }

    /// The interesting direction: arbitrary bytes, most of which are invalid
    /// hex, must be accepted or rejected exactly as the reference does.
    #[test]
    fn decode_arbitrary_matches_naive(input in vec(any::<u8>(), 0..256)) {
        prop_assert_eq!(base16ct::lower::decode_vec(&input).ok(), naive_decode(&input, Case::Lower));
        prop_assert_eq!(base16ct::upper::decode_vec(&input).ok(), naive_decode(&input, Case::Upper));
        prop_assert_eq!(base16ct::mixed::decode_vec(&input).ok(), naive_decode(&input, Case::Mixed));
    }

    /// Restricting the alphabet makes valid hex far more likely than `any::<u8>()`.
    #[test]
    fn decode_hex_alphabet_matches_naive(input in "[0-9a-fA-F]{0,256}") {
        let bytes = input.as_bytes();
        prop_assert_eq!(base16ct::lower::decode_vec(bytes).ok(), naive_decode(bytes, Case::Lower));
        prop_assert_eq!(base16ct::upper::decode_vec(bytes).ok(), naive_decode(bytes, Case::Upper));
        prop_assert_eq!(base16ct::mixed::decode_vec(bytes).ok(), naive_decode(bytes, Case::Mixed));
    }

    #[test]
    fn encoded_output_is_in_alphabet(input in vec(any::<u8>(), 0..256)) {
        let lower = base16ct::lower::encode_string(&input);
        let upper = base16ct::upper::encode_string(&input);
        prop_assert!(lower.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)));
        prop_assert!(upper.bytes().all(|b| b.is_ascii_digit() || (b'A'..=b'F').contains(&b)));
        prop_assert_eq!(lower.len(), base16ct::encoded_len(&input));
    }

    /// Injecting a non-hex byte anywhere must be rejected by every decoder.
    #[test]
    fn injected_invalid_byte_is_rejected(
        input in vec(any::<u8>(), 1..128),
        pos in any::<prop::sample::Index>(),
        bad in any::<u8>().prop_filter("must not be a hex digit", |b| !b.is_ascii_hexdigit()),
    ) {
        let mut hex = base16ct::lower::encode_string(&input).into_bytes();
        let pos = pos.index(hex.len());
        hex[pos] = bad;

        prop_assert_eq!(base16ct::lower::decode_vec(&hex), Err(Error::InvalidEncoding));
        prop_assert_eq!(base16ct::upper::decode_vec(&hex), Err(Error::InvalidEncoding));
        prop_assert_eq!(base16ct::mixed::decode_vec(&hex), Err(Error::InvalidEncoding));
    }

    /// The buffer and allocating APIs must agree.
    /// A strict decoder rejects the other alphabet.
    ///
    /// The interesting half is the converse: an encoding made only of digits is
    /// byte-identical in both cases, so it must still be accepted. Asserting
    /// that too is what stops the decoder from being trivially strict.
    #[test]
    fn strict_decoders_reject_only_the_other_alphabet(input in vec(any::<u8>(), 1..256)) {
        let lower = base16ct::lower::encode_string(&input);
        let upper = base16ct::upper::encode_string(&input);
        let mut buf = vec![0u8; input.len()];

        if lower == upper {
            prop_assert_eq!(
                base16ct::lower::decode(&upper, &mut buf),
                Ok(input.as_slice())
            );
            prop_assert_eq!(
                base16ct::upper::decode(&lower, &mut buf),
                Ok(input.as_slice())
            );
        } else {
            prop_assert_eq!(
                base16ct::lower::decode(&upper, &mut buf),
                Err(Error::InvalidEncoding)
            );
            prop_assert_eq!(
                base16ct::upper::decode(&lower, &mut buf),
                Err(Error::InvalidEncoding)
            );
        }
    }

    /// `mixed` accepts an arbitrary per-character mixture of the two alphabets,
    /// not merely uniformly-cased input.
    #[test]
    fn mixed_accepts_arbitrary_case_mixture(
        input in vec(any::<u8>(), 0..256),
        upcase in vec(any::<bool>(), 1..64),
    ) {
        let mut hex = base16ct::lower::encode_string(&input).into_bytes();

        for (byte, up) in hex.iter_mut().zip(upcase.iter().cycle()) {
            if *up {
                byte.make_ascii_uppercase();
            }
        }

        prop_assert_eq!(base16ct::mixed::decode_vec(&hex), Ok(input));
    }

    #[test]
    fn decode_vec_matches_decode_slice(input in vec(any::<u8>(), 0..256)) {
        let hex = base16ct::lower::encode_string(&input);
        let mut buf = vec![0u8; input.len()];
        let slice = base16ct::lower::decode(&hex, &mut buf).unwrap();
        let expected = base16ct::lower::decode_vec(&hex).unwrap();
        prop_assert_eq!(slice, expected.as_slice());
    }

    #[test]
    fn hex_display_matches_encode(input in vec(any::<u8>(), 0..256)) {
        prop_assert_eq!(
            format!("{:x}", base16ct::HexDisplay(&input)),
            base16ct::lower::encode_string(&input)
        );
        prop_assert_eq!(
            format!("{:X}", base16ct::HexDisplay(&input)),
            base16ct::upper::encode_string(&input)
        );
    }
}
