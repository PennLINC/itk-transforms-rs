# itk-transforms-rs

A pure-Rust reader for ITK composite spatial transforms — the
`*Composite.h5` files written by ANTs `antsRegistration`, plus the
`Insight Transform File V1.0` (`.txt`) format ITK and ANTs use for
plain-text affines.

Everything returned to the caller is in **RAS+ mm** so it composes
directly with NIfTI / ODX / TRX / MRtrix data without any further
sign-flipping. The LPS+ ↔ RAS+ conversion is centralised in one module
and validated against MRtrix3's `transformconvert itk_import` and the
nitransforms Python reference implementation.

## Why

ANTs writes affines and displacement fields in ITK's native LPS+
(Left-Posterior-Superior) coordinates inside HDF5. Most of the rest of
the neuroimaging stack (NIfTI, MRtrix conventions for diffusion,
ODX/TRX) works in RAS+ (Right-Anterior-Superior) mm. Mixing the two
without a single, well-tested boundary is how minus-sign bugs enter
pipelines.

## Quick start

```rust
use itk_transforms_rs::{read_itk, Result};

fn main() -> Result<()> {
    let chain = read_itk("Composite.h5".as_ref())?;

    // Map a point from fixed-image space to moving-image space (RAS+ mm),
    // matching antsApplyTransforms' pull-based semantics.
    let p_moved = chain.map_point([10.0, 20.0, 30.0]);
    println!("{:?}", p_moved);

    // 3×3 Jacobian (warps use central finite differences with the given step).
    let _j = chain.jacobian_at([10.0, 20.0, 30.0], 0.5);

    // Affine-only chains can be inverted exactly.
    if !chain.has_warp() {
        let _inv = chain.invert()?;
    }
    Ok(())
}
```

## Supported inputs

- ITK Composite HDF5 (`.h5`) — affine, rigid, similarity, and
  displacement-field components. Chains of multiple components are
  preserved in order.
- Insight Transform File V1.0 (`.txt`) — affine-only by spec; multiple
  `#Transform N` sections compose into a single chain.

## Out of scope

- Numerical inversion of displacement fields.
- Image resampling (point and Jacobian queries only).
- Writing transforms back out (the `into_itk_components` helpers exist
  for downstream writers but no file format is emitted here).

## Public API

- [`read_itk`](src/lib.rs) — dispatches on file extension.
- [`read_itk_h5`](src/itk_h5.rs), [`read_itk_txt`](src/itk_txt.rs) —
  format-specific readers.
- [`TransformChain`](src/chain.rs), [`TransformComponent`](src/chain.rs)
  — the composed transform.
- [`Affine3`](src/affine.rs), [`DisplacementField`](src/warp.rs),
  [`TargetGrid`](src/grid.rs) — component types.
- [`XfmError`](src/error.rs), `Result` — error type and alias.

## Testing

```sh
cargo test
```

Unit tests cover affine composition, LPS↔RAS round-trips, trilinear
displacement-field sampling, Jacobian chain rule, and parser edge
cases. Integration tests under `tests/` cross-validate against
fixtures from MRtrix3 (vendored under `tests/fixtures/mrtrix-affine/`,
MPL 2.0) and against canonical nitransforms output. Tests that depend
on the nitransforms fixture skip gracefully when the fixture is
absent.

## License

Dual-licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
