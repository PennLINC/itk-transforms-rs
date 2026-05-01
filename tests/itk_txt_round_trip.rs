//! Verifies the public [`read_itk_txt`] / [`read_itk`] API against the
//! same vendored mrtrix3 `transformconvert/` fixtures used by the
//! `mrtrix_affine_compat.rs` test. This isolates the *public* API surface
//! (anyone consuming the crate can rely on it) from the test-internal
//! parser used historically.

use std::fs;
use std::path::PathBuf;

use itk_transforms_rs::{read_itk, read_itk_txt, TransformComponent};
use nalgebra::Matrix4;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("mrtrix-affine")
        .join(name)
}

fn parse_mrtrix_matrix(path: &PathBuf) -> Matrix4<f64> {
    let body = fs::read_to_string(path).unwrap();
    let nums: Vec<f64> = body
        .split_whitespace()
        .filter_map(|tok| tok.parse::<f64>().ok())
        .collect();
    let mut m = Matrix4::identity();
    for r in 0..3 {
        for c in 0..4 {
            m[(r, c)] = nums[r * 4 + c];
        }
    }
    m
}

fn assert_close(got: &Matrix4<f64>, want: &Matrix4<f64>, label: &str) {
    let mut max = 0.0_f64;
    for r in 0..4 {
        for c in 0..4 {
            let d = (got[(r, c)] - want[(r, c)]).abs();
            if d > max {
                max = d;
            }
        }
    }
    assert!(
        max <= 1e-3,
        "{label}: max abs diff {max} > 1e-3\n got:\n{got}\nwant:\n{want}"
    );
}

#[test]
fn read_itk_txt_zero_center_matches_mrtrix_reference() {
    let chain = read_itk_txt(&fixture("affine_ants_zero.txt")).unwrap();
    assert_eq!(chain.components.len(), 1);
    let TransformComponent::Affine(a) = &chain.components[0] else {
        panic!("expected an affine");
    };
    let want = parse_mrtrix_matrix(&fixture("affine_mrtrix_zero.txt"));
    assert_close(&a.matrix, &want, "zero-center round-trip via public API");
}

#[test]
fn read_itk_txt_nonzero_center_matches_mrtrix_reference() {
    let chain = read_itk_txt(&fixture("affine_ants.txt")).unwrap();
    assert_eq!(chain.components.len(), 1);
    let TransformComponent::Affine(a) = &chain.components[0] else {
        panic!("expected an affine");
    };
    let want = parse_mrtrix_matrix(&fixture("affine_mrtrix.txt"));
    assert_close(
        &a.matrix,
        &want,
        "non-zero-center round-trip via public API",
    );
}

#[test]
fn read_itk_dispatches_by_extension() {
    let chain_via_dispatch = read_itk(&fixture("affine_ants_zero.txt")).unwrap();
    let chain_via_direct = read_itk_txt(&fixture("affine_ants_zero.txt")).unwrap();
    assert_eq!(
        chain_via_dispatch.components.len(),
        chain_via_direct.components.len()
    );
    let TransformComponent::Affine(a1) = &chain_via_dispatch.components[0] else {
        panic!()
    };
    let TransformComponent::Affine(a2) = &chain_via_direct.components[0] else {
        panic!()
    };
    assert_eq!(a1.matrix, a2.matrix);
}

#[test]
fn read_itk_rejects_unknown_extension() {
    use std::io::Write;
    let mut f = tempfile::Builder::new()
        .suffix(".bogus")
        .tempfile()
        .unwrap();
    writeln!(f, "garbage").unwrap();
    f.flush().unwrap();
    let err = read_itk(f.path()).unwrap_err();
    assert!(matches!(
        err,
        itk_transforms_rs::XfmError::InvalidFile { .. }
    ));
}
