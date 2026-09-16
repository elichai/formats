use crate::{Error, backends::UPPER, decode_inner, encode_inner};
#[cfg(feature = "alloc")]
use crate::{String, Vec, decoded_len, encoded_len};

/// Decode an upper Base16 (hex) string into the provided destination buffer.
pub fn decode(src: impl AsRef<[u8]>, dst: &mut [u8]) -> Result<&[u8], Error> {
    decode_inner::<UPPER>(src.as_ref(), dst)
}

/// Decode an upper Base16 (hex) string into a byte vector.
#[cfg(feature = "alloc")]
pub fn decode_vec(input: impl AsRef<[u8]>) -> Result<Vec<u8>, Error> {
    let mut output = vec![0u8; decoded_len(input.as_ref())?];
    decode(input, &mut output)?;
    Ok(output)
}

/// Encode the input byte slice as upper Base16.
///
/// Writes the result into the provided destination slice, returning an
/// ASCII-encoded upper Base16 (hex) string value.
pub fn encode<'a>(src: &[u8], dst: &'a mut [u8]) -> Result<&'a [u8], Error> {
    encode_inner(src, dst, true)
}

/// Encode input byte slice into a [`&str`] containing upper Base16 (hex).
pub fn encode_str<'a>(src: &[u8], dst: &'a mut [u8]) -> Result<&'a str, Error> {
    // SAFETY: `encode` only ever writes ASCII hex digits, which are valid UTF-8.
    #[allow(unsafe_code)]
    encode(src, dst).map(|r| unsafe { core::str::from_utf8_unchecked(r) })
}

/// Encode input byte slice into a [`String`] containing upper Base16 (hex).
///
/// # Panics
/// If `input` length is greater than `usize::MAX/2`.
#[cfg(feature = "alloc")]
pub fn encode_string(input: &[u8]) -> String {
    let elen = encoded_len(input);
    let mut dst = vec![0u8; elen];
    let res = encode(input, &mut dst).expect("dst length is correct");

    debug_assert_eq!(elen, res.len());
    // SAFETY: `encode` only ever writes ASCII hex digits, which are valid UTF-8.
    #[allow(unsafe_code)]
    unsafe {
        String::from_utf8_unchecked(dst)
    }
}
