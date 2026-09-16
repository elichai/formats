# [RustCrypto]: Constant-Time Base16 (hexadecimal)

[![crate][crate-image]][crate-link]
[![Docs][docs-image]][docs-link]
[![Build Status][build-image]][build-link]
![Apache2/MIT licensed][license-image]
![Rust Version][rustc-image]
[![Project Chat][chat-image]][chat-link]

Pure Rust implementation of Base16 ([RFC 4648]).

Implements lower and upper case Base16 variants without data-dependent branches
or lookup  tables, thereby providing portable "best effort" constant-time
operation.

Supports `no_std` environments and avoids heap allocations in the core API
(but also provides optional `alloc` support for convenience).

[Documentation][docs-link]

## Backends

On `x86`/`x86_64` the widest supported SIMD tier is selected at runtime via
[`cpufeatures`], falling back to a portable implementation.

On `aarch64` and `wasm32` the choice is made at compile time from the target
features, with no detection involved: `neon` is enabled by default on
`aarch64`, and WebAssembly has no runtime feature query, so `simd128` support
is a property of how the module was built. Enable it with
`-Ctarget-feature=+simd128`.

No configuration is needed in any of these cases.

The lookup tables used by the SIMD backends are held in registers rather than
memory, so they introduce no data-dependent memory access and the crate's
"best effort" constant-time property is preserved.

This is checked rather than asserted: `ct-tests/` runs the backends under
Valgrind's memcheck with the inputs marked undefined, so any branch or address
derived from input *contents* is reported. Note that memcheck tracks
definedness through x86 vector registers but not through `aarch64` NEON
registers, so the check is stronger on x86; see that crate's documentation for
exactly what each architecture covers.

A backend can be pinned at compile time with the `base16ct_backend`
configuration flag:

- `soft`: portable implementation only. Excludes the SIMD backends from the
  build entirely, on every architecture.
- `x86-ssse3`: SSSE3. Requires the `ssse3` target feature.
- `x86-avx2`: AVX2. Requires the `avx2` target feature.
- `x86-avx512`: AVX-512. Requires the `avx512bw` and `avx512vbmi` target
  features. Decoding needs only `avx512bw`, but encoding uses `vpermi2b`, so
  pinning the tier requires both; under autodetection the two are detected
  separately and decode will use AVX-512 on CPUs where encode cannot.

There are no values for NEON or `simd128`: those architectures have a single
tier each, so `soft` against the default is already the only choice available.

Pinning a backend that requires an unavailable target feature is a compile
error rather than a runtime fault. Set the flag through `RUSTFLAGS`:

```sh
RUSTFLAGS='--cfg base16ct_backend="soft"' cargo build
RUSTFLAGS='-Ctarget-feature=+avx2 --cfg base16ct_backend="x86-avx2"' cargo build
```

or in `.cargo/config.toml`. Note this is a configuration flag rather than a
Cargo feature, so that an unrelated crate in the dependency graph cannot change
which backend you build.

[`cpufeatures`]: https://docs.rs/cpufeatures

## Minimum Supported Rust Version (MSRV) Policy

MSRV increases are not considered breaking changes and can happen in patch releases.

The crate MSRV accounts for all supported targets and crate feature combinations, excluding
explicitly unstable features.

## License

Licensed under either of:

 * [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0)
 * [MIT license](http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.

[//]: # (badges)

[crate-image]: https://img.shields.io/crates/v/base16ct
[crate-link]: https://crates.io/crates/base16ct
[docs-image]: https://docs.rs/base16ct/badge.svg
[docs-link]: https://docs.rs/base16ct/
[build-image]: https://github.com/RustCrypto/formats/actions/workflows/base16ct.yml/badge.svg
[build-link]: https://github.com/RustCrypto/formats/actions/workflows/base16ct.yml
[license-image]: https://img.shields.io/badge/license-Apache2.0/MIT-blue.svg
[rustc-image]: https://img.shields.io/badge/rustc-1.89+-blue.svg
[chat-image]: https://img.shields.io/badge/zulip-join_chat-blue.svg
[chat-link]: https://rustcrypto.zulipchat.com/#narrow/stream/300570-formats

[//]: # (links)

[RustCrypto]: https://github.com/rustcrypto
[RFC 4648]: https://tools.ietf.org/html/rfc4648
[Util::Lookup]: https://arxiv.org/pdf/2108.04600.pdf
