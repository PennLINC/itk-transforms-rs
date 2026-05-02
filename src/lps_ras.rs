//! Single source of truth for the LPS+ ↔ RAS+ flip.
//!
//! ITK works in LPS+ (Left-Posterior-Superior); NIfTI/ODX/TRX/MRtrix work in
//! RAS+ (Right-Anterior-Superior). The world-frame conversion matrix is
//! `LPS = diag(-1, -1, 1, 1)`, applied differently depending on what is
//! being converted:
//!
//! - **Affine transform** (point in frame F → point in frame F): the
//!   transform itself needs both *input* and *output* re-expressed in the
//!   target frame, so `M_ras = LPS · M_itk · LPS` (a similarity sandwich,
//!   self-inverse). Use [`affine_itk_to_ras`].
//! - **Grid affine** (voxel index → world point): voxel indices are
//!   coordinate-system-agnostic; only the *output* world point needs the
//!   flip, so `M_ras = LPS · M_itk` (left-multiply only). Apply [`lps4`]
//!   directly at the call site.
//!
//! Every site in this crate that crosses the ITK boundary calls into this
//! module — there are no scattered `-1`s anywhere else.

use nalgebra::Matrix4;

/// `diag(-1, -1, 1, 1)`: flips x and y, keeps z, in homogeneous coordinates.
#[inline]
pub(crate) fn lps4() -> Matrix4<f64> {
    let mut m = Matrix4::identity();
    m[(0, 0)] = -1.0;
    m[(1, 1)] = -1.0;
    m
}

/// Convert a 4×4 affine from ITK (LPS+) to RAS+.
///
/// `M_ras = LPS · M_itk · LPS`. Self-inverse: applying twice returns the input.
#[inline]
pub(crate) fn affine_itk_to_ras(itk: &Matrix4<f64>) -> Matrix4<f64> {
    let lps = lps4();
    lps * itk * lps
}

/// Convert a 4×4 affine from RAS+ to ITK (LPS+).
///
/// Identical to [`affine_itk_to_ras`] (the LPS flip is its own inverse) but
/// kept as a separate name for clarity at call sites.
#[inline]
pub(crate) fn affine_ras_to_itk(ras: &Matrix4<f64>) -> Matrix4<f64> {
    affine_itk_to_ras(ras)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Matrix4;

    #[test]
    fn lps_is_self_inverse() {
        let lps = lps4();
        assert_eq!(lps * lps, Matrix4::identity());
    }

    #[test]
    fn round_trip_identity_affine() {
        let itk = Matrix4::<f64>::identity();
        assert_eq!(affine_itk_to_ras(&itk), Matrix4::identity());
    }

    #[test]
    fn round_trip_translation() {
        // ITK translation [1, 2, 3] in LPS == translation [-1, -2, 3] in RAS.
        let mut itk = Matrix4::<f64>::identity();
        itk[(0, 3)] = 1.0;
        itk[(1, 3)] = 2.0;
        itk[(2, 3)] = 3.0;
        let ras = affine_itk_to_ras(&itk);
        assert_eq!(ras[(0, 3)], -1.0);
        assert_eq!(ras[(1, 3)], -2.0);
        assert_eq!(ras[(2, 3)], 3.0);
        // And it's an involution.
        assert_eq!(affine_ras_to_itk(&ras), itk);
    }

    #[test]
    fn round_trip_rotation_about_z() {
        // 90° rotation about z: same matrix in either convention but with the
        // x and y rows/cols flipped through the LPS sandwich.
        let theta = std::f64::consts::FRAC_PI_2;
        let mut itk = Matrix4::<f64>::identity();
        let (s, c) = (theta.sin(), theta.cos());
        itk[(0, 0)] = c;
        itk[(0, 1)] = -s;
        itk[(1, 0)] = s;
        itk[(1, 1)] = c;
        let ras = affine_itk_to_ras(&itk);
        // Rz(90°) is invariant under x,y sign flip applied on both sides.
        assert!((ras[(0, 0)] - c).abs() < 1e-12);
        assert!((ras[(0, 1)] + s).abs() < 1e-12);
        assert!((ras[(1, 0)] - s).abs() < 1e-12);
        assert!((ras[(1, 1)] - c).abs() < 1e-12);
    }
}
