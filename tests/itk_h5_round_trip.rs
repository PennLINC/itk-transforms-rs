//! Integration tests for the ITK Composite.h5 reader.
//!
//! Cross-checks against nitransforms' Python implementation: the canonical
//! `affine-antsComposite.h5` fixture vendored under
//! `tests/fixtures/nitransforms/` is loaded, converted to RAS+, and the
//! resulting matrix is asserted against values computed independently in
//! Python (see comments in source).

use std::path::PathBuf;

use itk_transforms_rs::{read_itk_h5, TransformChain, TransformComponent};

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/nitransforms/affine-antsComposite.h5")
}

#[test]
fn parses_canonical_ants_composite_fixture() {
    let path = fixture_path();
    let chain: TransformChain = read_itk_h5(&path).expect("reader should succeed");
    assert_eq!(
        chain.components.len(),
        1,
        "expected exactly one (non-wrapper) component"
    );
    let TransformComponent::Affine(a) = &chain.components[0] else {
        panic!("expected an AffineTransform");
    };

    // Independently computed in Python (see docstring above):
    //   ras = LPS · T(+c) · [[M | t] / [0 0 0 1]] · T(-c) · LPS
    // for the parameters in the fixture.
    let expected: [[f64; 4]; 4] = [
        [
            0.887202094795028,
            -0.046512085061111,
            0.051773924135844,
            0.814590250415989,
        ],
        [
            0.067795705592731,
            0.897189913445307,
            -0.131372358536528,
            18.077_925_472_451_3,
        ],
        [
            -0.023175351829446,
            0.140799187585230,
            0.857501368426337,
            18.471859867162042,
        ],
        [0.0, 0.0, 0.0, 1.0],
    ];

    for (r, row) in expected.iter().enumerate() {
        for (c, &want) in row.iter().enumerate() {
            let got = a.matrix[(r, c)];
            assert!(
                (got - want).abs() < 1e-12,
                "matrix[{r}][{c}]: got {got}, want {want}",
            );
        }
    }

    // Spot-check a transformed point against Python.
    let p = chain.map_point([10.0, 20.0, 30.0]);
    let want = [
        10.309_587_221_219_35,
        32.758_510_041_188_9,
        46.781_131_153_362_296,
    ];
    for (i, &want_i) in want.iter().enumerate() {
        assert!(
            (p[i] - want_i).abs() < 1e-9,
            "point[{i}]: got {} want {}",
            p[i],
            want_i
        );
    }
}

#[test]
fn missing_file_returns_error() {
    let p = PathBuf::from("/nonexistent/path/does/not/exist.h5");
    assert!(read_itk_h5(&p).is_err());
}
