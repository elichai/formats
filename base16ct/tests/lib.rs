//! Integration tests.

/// Hexadecimal test vectors
struct HexVector {
    /// Raw bytes
    raw: &'static [u8],
    /// Lower hex encoded
    lower_hex: &'static [u8],
    /// Upper hex encoded
    upper_hex: &'static [u8],
}

const HEX_TEST_VECTORS: &[HexVector] = &[
    HexVector {
        raw: b"",
        lower_hex: b"",
        upper_hex: b"",
    },
    HexVector {
        raw: b"\0",
        lower_hex: b"00",
        upper_hex: b"00",
    },
    HexVector {
        raw: b"***",
        lower_hex: b"2a2a2a",
        upper_hex: b"2A2A2A",
    },
    HexVector {
        raw: b"\x01\x02\x03\x04",
        lower_hex: b"01020304",
        upper_hex: b"01020304",
    },
    HexVector {
        raw: b"\xAD\xAD\xAD\xAD\xAD",
        lower_hex: b"adadadadad",
        upper_hex: b"ADADADADAD",
    },
    HexVector {
        raw: b"\xFF\xFF\xFF\xFF\xFF",
        lower_hex: b"ffffffffff",
        upper_hex: b"FFFFFFFFFF",
    },
];

#[test]
fn lower_encode() {
    for vector in HEX_TEST_VECTORS {
        // 10 is the size of the largest encoded test vector
        let mut buf = [0u8; 10];
        let out = base16ct::lower::encode(vector.raw, &mut buf).unwrap();
        assert_eq!(vector.lower_hex, out);
    }
}

#[test]
fn lower_decode() {
    for vector in HEX_TEST_VECTORS {
        // 5 is the size of the largest decoded test vector
        let mut buf = [0u8; 5];
        let out = base16ct::lower::decode(vector.lower_hex, &mut buf).unwrap();
        assert_eq!(vector.raw, out);
    }
}

#[test]
fn lower_reject_odd_size_input() {
    let mut out = [0u8; 3];
    assert_eq!(
        Err(base16ct::Error::InvalidLength),
        base16ct::lower::decode(b"12345", &mut out),
    )
}

#[test]
fn upper_encode() {
    for vector in HEX_TEST_VECTORS {
        // 10 is the size of the largest encoded test vector
        let mut buf = [0u8; 10];
        let out = base16ct::upper::encode(vector.raw, &mut buf).unwrap();
        assert_eq!(vector.upper_hex, out);
    }
}

#[test]
fn upper_decode() {
    for vector in HEX_TEST_VECTORS {
        // 5 is the size of the largest decoded test vector
        let mut buf = [0u8; 5];
        let out = base16ct::upper::decode(vector.upper_hex, &mut buf).unwrap();
        assert_eq!(vector.raw, out);
    }
}

#[test]
fn upper_reject_odd_size_input() {
    let mut out = [0u8; 3];
    assert_eq!(
        Err(base16ct::Error::InvalidLength),
        base16ct::upper::decode(b"12345", &mut out),
    )
}

#[test]
fn mixed_decode() {
    for vector in HEX_TEST_VECTORS {
        // 5 is the size of the largest decoded test vector
        let mut buf = [0u8; 5];
        let out = base16ct::mixed::decode(vector.upper_hex, &mut buf).unwrap();
        assert_eq!(vector.raw, out);
        let out = base16ct::mixed::decode(vector.lower_hex, &mut buf).unwrap();
        assert_eq!(vector.raw, out);
    }
}

#[test]
fn mixed_reject_odd_size_input() {
    let mut out = [0u8; 3];
    assert_eq!(
        Err(base16ct::Error::InvalidLength),
        base16ct::mixed::decode(b"12345", &mut out),
    )
}

#[test]
#[cfg(feature = "alloc")]
fn encode_and_decode_various_lengths() {
    let data = [b'X'; 64];

    for i in 0..data.len() {
        let encoded = base16ct::lower::encode_string(&data[..i]);
        let decoded = base16ct::lower::decode_vec(encoded).unwrap();
        assert_eq!(decoded.as_slice(), &data[..i]);

        let encoded = base16ct::upper::encode_string(&data[..i]);
        let decoded = base16ct::upper::decode_vec(encoded).unwrap();
        assert_eq!(decoded.as_slice(), &data[..i]);

        let encoded = base16ct::lower::encode_string(&data[..i]);
        let decoded = base16ct::mixed::decode_vec(encoded).unwrap();
        assert_eq!(decoded.as_slice(), &data[..i]);

        let encoded = base16ct::upper::encode_string(&data[..i]);
        let decoded = base16ct::mixed::decode_vec(encoded).unwrap();
        assert_eq!(decoded.as_slice(), &data[..i]);
    }
}

#[test]
fn hex_display_upper() {
    for vector in HEX_TEST_VECTORS {
        let hex = format!("{:X}", base16ct::HexDisplay(vector.raw));
        assert_eq!(hex.as_bytes(), vector.upper_hex);
    }
}

#[test]
fn hex_display_lower() {
    for vector in HEX_TEST_VECTORS {
        let hex = format!("{:x}", base16ct::HexDisplay(vector.raw));
        assert_eq!(hex.as_bytes(), vector.lower_hex);
    }
}

// The tests below cover cases the original suite left untested: `InvalidEncoding`
// was never exercised at all, and neither was the case-strictness of the split
// `lower`/`upper` decoders.

use base16ct::Error;

/// Kept example-based rather than moved to `proptests.rs`: these are the bytes
/// immediately outside each valid range (`0/`, `0:`, `0@`, `0G`, `` 0` ``,
/// `0g`), which random generation samples only rarely. The general "any
/// non-hex byte is rejected" property lives in `proptests.rs`.
#[test]
fn reject_non_hex_characters() {
    const NON_HEX: &[&[u8]] = &[
        b"zz",
        b"0g",
        b"g0",
        b"\x00\x00",
        b"\xff\xff",
        b"  ",
        b"0/",
        b"0:",
        b"0@",
        b"0G",
        b"0`",
        b"0g",
        b"00zz",
        b"zz00",
    ];

    let mut buf = [0u8; 8];

    for input in NON_HEX {
        assert_eq!(
            base16ct::lower::decode(input, &mut buf),
            Err(Error::InvalidEncoding)
        );
        assert_eq!(
            base16ct::upper::decode(input, &mut buf),
            Err(Error::InvalidEncoding)
        );
        assert_eq!(
            base16ct::mixed::decode(input, &mut buf),
            Err(Error::InvalidEncoding)
        );
    }
}

#[test]
fn encode_rejects_undersized_dst() {
    let mut buf = [0u8; 3];
    assert_eq!(
        base16ct::lower::encode(b"\x01\x02", &mut buf),
        Err(Error::InvalidLength)
    );
    assert_eq!(
        base16ct::upper::encode(b"\x01\x02", &mut buf),
        Err(Error::InvalidLength)
    );
    // Exactly-sized destination is accepted.
    let mut exact = [0u8; 4];
    assert_eq!(
        base16ct::lower::encode(b"\x01\x02", &mut exact),
        Ok(&b"0102"[..])
    );
}

#[test]
fn decode_rejects_undersized_dst() {
    let mut buf = [0u8; 1];
    assert_eq!(
        base16ct::lower::decode(b"0102", &mut buf),
        Err(Error::InvalidLength)
    );
    assert_eq!(
        base16ct::upper::decode(b"0102", &mut buf),
        Err(Error::InvalidLength)
    );
    assert_eq!(
        base16ct::mixed::decode(b"0102", &mut buf),
        Err(Error::InvalidLength)
    );
}

#[test]
fn decoded_len_rejects_odd_lengths() {
    assert_eq!(base16ct::decoded_len(b""), Ok(0));
    assert_eq!(base16ct::decoded_len(b"00"), Ok(1));
    assert_eq!(base16ct::decoded_len(b"0000"), Ok(2));
    assert_eq!(base16ct::decoded_len(b"0"), Err(Error::InvalidLength));
    assert_eq!(base16ct::decoded_len(b"000"), Err(Error::InvalidLength));
}

#[test]
fn encoded_len_doubles() {
    assert_eq!(base16ct::encoded_len(b""), 0);
    assert_eq!(base16ct::encoded_len(b"\x00"), 2);
    assert_eq!(base16ct::encoded_len(b"\x00\x01\x02"), 6);
}

/// An invalid byte must be rejected wherever it appears, not just early on.
#[test]
fn reject_invalid_byte_at_every_position() {
    let raw = [0x12u8, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0];
    let mut hex = [0u8; 16];
    let mut buf = [0u8; 8];

    base16ct::lower::encode(&raw, &mut hex)
        .unwrap_or_else(|e| panic!("exactly-sized destination rejected: {e}"));

    for pos in 0..hex.len() {
        let saved = hex[pos];
        hex[pos] = b'G';
        assert_eq!(
            base16ct::lower::decode(hex, &mut buf),
            Err(Error::InvalidEncoding),
            "invalid byte at position {pos} accepted"
        );
        hex[pos] = saved;
    }

    // Restoring every byte leaves the input valid again.
    assert_eq!(base16ct::lower::decode(hex, &mut buf), Ok(&raw[..]));
}
