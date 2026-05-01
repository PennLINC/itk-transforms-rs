//! Tier-1 verification: the LPS↔RAS conversion in [`Affine3::from_itk_components`]
//! must produce the same RAS+ matrix that MRtrix3's `transformconvert
//! itk_import` produces. Cross-checked against the upstream
//! `transformconvert/itk_ants` test (assertion: `testing_diff_matrix -abs 1e-3`).
//!
//! Fixture provenance is in `fixtures/mrtrix-affine/README.md`.

use std::fs;
use std::path::PathBuf;

use itk_transforms_rs::Affine3;
use nalgebra::{Matrix3, Matrix4, Vector3};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("mrtrix-affine")
        .join(name)
}

/// Parse an ITK "Insight Transform File V1.0" textual transform. Returns
/// `(matrix3, translation, center)`. We tolerate either trailing or
/// embedded blank lines and arbitrary whitespace separators.
fn parse_itk_txt(path: &PathBuf) -> (Matrix3<f64>, Vector3<f64>, Vector3<f64>) {
    let body = fs::read_to_string(path).unwrap_or_else(|e| panic!("read {:?}: {e}", path));

    let mut params: Option<Vec<f64>> = None;
    let mut fixed: Option<Vec<f64>> = None;

    for line in body.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("Parameters:") {
            params = Some(parse_floats(rest));
        } else if let Some(rest) = line.strip_prefix("FixedParameters:") {
            fixed = Some(parse_floats(rest));
        }
    }

    let p = params.expect("Parameters: line missing");
    let f = fixed.expect("FixedParameters: line missing");
    assert_eq!(
        p.len(),
        12,
        "expected 12 ITK transform parameters, got {}",
        p.len()
    );
    assert_eq!(
        f.len(),
        3,
        "expected 3 ITK fixed parameters, got {}",
        f.len()
    );

    let m = Matrix3::new(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]);
    let t = Vector3::new(p[9], p[10], p[11]);
    let c = Vector3::new(f[0], f[1], f[2]);
    (m, t, c)
}

/// Parse an MRtrix3 4×4 RAS+ matrix file (whitespace-separated; one row
/// per line, last row optional/`0 0 0 1`).
fn parse_mrtrix_matrix(path: &PathBuf) -> Matrix4<f64> {
    let body = fs::read_to_string(path).unwrap_or_else(|e| panic!("read {:?}: {e}", path));
    let nums = parse_floats(&body);
    assert!(
        nums.len() == 12 || nums.len() == 16,
        "MRtrix matrix needs 12 or 16 floats, got {}",
        nums.len()
    );
    let mut m = Matrix4::identity();
    for r in 0..3 {
        for c in 0..4 {
            m[(r, c)] = nums[r * 4 + c];
        }
    }
    m
}

fn parse_floats(s: &str) -> Vec<f64> {
    s.split_whitespace()
        .filter_map(|tok| tok.parse::<f64>().ok())
        .collect()
}

/// Tolerance: MRtrix3 itself uses 1e-3 absolute (testing_diff_matrix). We
/// require that our recovered matrix is at most that far from theirs in
/// any element. In practice the agreement is ~1e-7 — both implementations
/// run the same arithmetic in `f64`.
const TOL: f64 = 1e-3;

fn assert_matrices_close(got: &Matrix4<f64>, want: &Matrix4<f64>, label: &str) {
    let mut max_diff = 0.0_f64;
    for r in 0..4 {
        for c in 0..4 {
            let d = (got[(r, c)] - want[(r, c)]).abs();
            if d > max_diff {
                max_diff = d;
            }
        }
    }
    assert!(
        max_diff <= TOL,
        "{label}: max element diff {max_diff} > {TOL}\n got:\n{}\nwant:\n{}",
        got,
        want,
    );
}

#[test]
fn ras_conversion_matches_mrtrix_with_zero_center() {
    let (m, t, c) = parse_itk_txt(&fixture("affine_ants_zero.txt"));
    assert_eq!(c, Vector3::zeros(), "fixture should have zero center");
    let want = parse_mrtrix_matrix(&fixture("affine_mrtrix_zero.txt"));
    let got = Affine3::from_itk_components(m, t, c).matrix;
    assert_matrices_close(&got, &want, "zero-center LPS sandwich");
}

#[test]
fn ras_conversion_matches_mrtrix_with_nonzero_center() {
    let (m, t, c) = parse_itk_txt(&fixture("affine_ants.txt"));
    assert!(c.norm() > 1.0, "fixture should have a nonzero center");
    let want = parse_mrtrix_matrix(&fixture("affine_mrtrix.txt"));
    let got = Affine3::from_itk_components(m, t, c).matrix;
    assert_matrices_close(&got, &want, "non-zero-center T(+c)·M·T(-c) sandwich");
}

#[test]
fn applying_affine_matches_mrtrix_pointwise() {
    // Spot-check: take the canonical (non-zero-center) pair and apply both
    // matrices to a couple of arbitrary RAS+ points; the results must
    // agree to 1e-4 mm. This catches any sign-flip in the matrix even if
    // the elementwise diff above is too lenient on a particular cell.
    let (m, t, c) = parse_itk_txt(&fixture("affine_ants.txt"));
    let mrtrix = parse_mrtrix_matrix(&fixture("affine_mrtrix.txt"));
    let ours = Affine3::from_itk_components(m, t, c);

    for p in [[10.0, -5.0, 32.0], [0.0, 0.0, 0.0], [-3.0, 50.0, 7.5]] {
        let got = ours.apply(p);
        let v = mrtrix * nalgebra::Vector4::new(p[0], p[1], p[2], 1.0);
        let want = [v[0], v[1], v[2]];
        for i in 0..3 {
            assert!(
                (got[i] - want[i]).abs() < 1e-4,
                "point {p:?} axis {i}: got {} want {}",
                got[i],
                want[i],
            );
        }
    }
}
