use crate::{Error, backends::MIXED, decode_inner};
#[cfg(feature = "alloc")]
use crate::{Vec, decoded_len};

/// Decode a mixed Base16 (hex) string into the provided destination buffer.
pub fn decode(src: impl AsRef<[u8]>, dst: &mut [u8]) -> Result<&[u8], Error> {
    decode_inner::<MIXED>(src.as_ref(), dst)
}

/// Decode a mixed Base16 (hex) string into a byte vector.
#[cfg(feature = "alloc")]
pub fn decode_vec(input: impl AsRef<[u8]>) -> Result<Vec<u8>, Error> {
    let mut output = vec![0u8; decoded_len(input.as_ref())?];
    decode(input, &mut output)?;
    Ok(output)
}
