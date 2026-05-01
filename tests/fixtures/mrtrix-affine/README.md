# MRtrix3 transformconvert fixtures

These four ASCII fixtures are vendored verbatim from the MRtrix3 binary
test-data repository:

- Repo: <https://github.com/mattcieslak/test_data>
- Tag:  `e3a85f94bf79b0556d940c9ffde3899eb86d7dd8`
- Path: `transformconvert/{affine_ants,affine_ants_zero,affine_mrtrix,affine_mrtrix_zero}.txt`

They are used by MRtrix3's own `transformconvert/itk_ants` test, which
exercises the LPS↔RAS conversion that the present crate also implements.
A successful test here means our `Affine3::from_itk_components` produces a
RAS+ matrix that matches MRtrix3's reference within `1e-3` (the same
tolerance MRtrix3 uses, via `testing_diff_matrix -abs 1e-3`).

| File                       | Format                       | Center of rotation |
|----------------------------|------------------------------|--------------------|
| `affine_ants.txt`          | ITK Insight V1.0 (LPS+)      | `(2.14, 4.70, 24.39)` |
| `affine_mrtrix.txt`        | MRtrix3 4×4 RAS+ matrix      | n/a (origin-centered) |
| `affine_ants_zero.txt`     | ITK Insight V1.0 (LPS+)      | `(0, 0, 0)` |
| `affine_mrtrix_zero.txt`   | MRtrix3 4×4 RAS+ matrix      | n/a (origin-centered) |

The non-zero-center pair exercises the full
`T(+c) · M · T(−c)` reconstruction; the zero-center pair isolates the bare
LPS sandwich.

Upstream MRtrix3 is licensed under the Mozilla Public License 2.0; these
ASCII parameter dumps carry no copyrightable expression beyond the numeric
content but are nevertheless included here under fair-use conventions for
test interoperability. Original source kept intact — no modifications.
