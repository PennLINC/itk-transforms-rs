//! 3D affine transform — a homogeneous 4×4 matrix in RAS+ mm.

use nalgebra::{Matrix3, Matrix4, Vector3, Vector4};

use crate::error::{Result, XfmError};
use crate::lps_ras::{affine_itk_to_ras, affine_ras_to_itk};

/// 3D affine transform stored as a row-major 4×4 in RAS+ mm.
///
/// `apply(p) = (matrix · [p; 1]).xyz`.
///
/// # Examples
///
/// ```
/// use itk_transforms_rs::Affine3;
/// use nalgebra::{Matrix3, Vector3};
///
/// // Pure translation in ITK-LPS+ becomes a sign-flipped translation in RAS+.
/// let a = Affine3::from_itk_components(
///     Matrix3::identity(),
///     Vector3::new(1.0, 2.0, 3.0),
///     Vector3::zeros(),
/// );
/// assert_eq!(a.apply([0.0, 0.0, 0.0]), [-1.0, -2.0, 3.0]);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Affine3 {
    /// Row-major 4×4 homogeneous matrix in RAS+ mm.
    pub matrix: Matrix4<f64>,
}

impl Affine3 {
    /// Identity transform.
    pub fn identity() -> Self {
        Self {
            matrix: Matrix4::identity(),
        }
    }

    /// Build from a 4×4 RAS+ matrix.
    pub fn from_matrix(matrix: Matrix4<f64>) -> Self {
        Self { matrix }
    }

    /// Build from a row-major `[[f64; 4]; 4]`.
    pub fn from_rows(rows: [[f64; 4]; 4]) -> Self {
        Self {
            matrix: Matrix4::from_row_slice(&[
                rows[0][0], rows[0][1], rows[0][2], rows[0][3], rows[1][0], rows[1][1], rows[1][2],
                rows[1][3], rows[2][0], rows[2][1], rows[2][2], rows[2][3], rows[3][0], rows[3][1],
                rows[3][2], rows[3][3],
            ]),
        }
    }

    /// Build a RAS+ affine from an ITK affine in LPS+.
    ///
    /// ITK serialises affines as a 3×3 matrix `M`, a 3-vector translation
    /// `t`, and a 3-vector center of rotation `c`. The full mapping is
    /// `p_out = M · (p_in − c) + c + t`, i.e. `T(+c) · [M | t] · T(−c)` in
    /// homogeneous form. After assembling that, we convert to RAS+ via the
    /// LPS sandwich.
    pub fn from_itk_components(
        matrix: Matrix3<f64>,
        translation: Vector3<f64>,
        center: Vector3<f64>,
    ) -> Self {
        let mut t_neg = Matrix4::identity();
        t_neg[(0, 3)] = -center[0];
        t_neg[(1, 3)] = -center[1];
        t_neg[(2, 3)] = -center[2];

        let mut t_pos = Matrix4::identity();
        t_pos[(0, 3)] = center[0];
        t_pos[(1, 3)] = center[1];
        t_pos[(2, 3)] = center[2];

        let mut core = Matrix4::identity();
        core.fixed_view_mut::<3, 3>(0, 0).copy_from(&matrix);
        core[(0, 3)] = translation[0];
        core[(1, 3)] = translation[1];
        core[(2, 3)] = translation[2];

        let itk = t_pos * core * t_neg;
        Self {
            matrix: affine_itk_to_ras(&itk),
        }
    }

    /// Inverse mapping into ITK component form (used by writers).
    /// Returns `(matrix, translation)`; center is taken to be zero.
    pub fn into_itk_components(&self) -> (Matrix3<f64>, Vector3<f64>) {
        let itk = affine_ras_to_itk(&self.matrix);
        let m = itk.fixed_view::<3, 3>(0, 0).into_owned();
        let t = Vector3::new(itk[(0, 3)], itk[(1, 3)], itk[(2, 3)]);
        (m, t)
    }

    /// Apply to a 3D point.
    #[inline]
    pub fn apply(&self, p: [f64; 3]) -> [f64; 3] {
        let v = self.matrix * Vector4::new(p[0], p[1], p[2], 1.0);
        [v[0], v[1], v[2]]
    }

    /// The 3×3 linear part (rotation + scale + shear).
    #[inline]
    pub fn linear(&self) -> Matrix3<f64> {
        self.matrix.fixed_view::<3, 3>(0, 0).into_owned()
    }

    /// The 3-vector translation.
    #[inline]
    pub fn translation(&self) -> Vector3<f64> {
        Vector3::new(
            self.matrix[(0, 3)],
            self.matrix[(1, 3)],
            self.matrix[(2, 3)],
        )
    }

    /// Invert. Returns an error if the matrix is singular.
    pub fn try_inverse(&self) -> Result<Self> {
        self.matrix
            .try_inverse()
            .map(Self::from_matrix)
            .ok_or(XfmError::SingularMatrix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_apply() {
        let a = Affine3::identity();
        assert_eq!(a.apply([1.0, 2.0, 3.0]), [1.0, 2.0, 3.0]);
    }

    #[test]
    fn translation_apply() {
        let mut m = Matrix4::identity();
        m[(0, 3)] = 10.0;
        m[(1, 3)] = 20.0;
        m[(2, 3)] = 30.0;
        let a = Affine3::from_matrix(m);
        assert_eq!(a.apply([1.0, 2.0, 3.0]), [11.0, 22.0, 33.0]);
    }

    #[test]
    fn inverse_translation() {
        let mut m = Matrix4::identity();
        m[(0, 3)] = 5.0;
        let a = Affine3::from_matrix(m);
        let inv = a.try_inverse().unwrap();
        assert_eq!(inv.apply([5.0, 0.0, 0.0]), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn itk_translation_round_trip() {
        // ITK translates by (1, 2, 3) in LPS — that should land RAS coords
        // shifted by (-1, -2, 3).
        let a = Affine3::from_itk_components(
            Matrix3::identity(),
            Vector3::new(1.0, 2.0, 3.0),
            Vector3::zeros(),
        );
        assert_eq!(a.apply([0.0, 0.0, 0.0]), [-1.0, -2.0, 3.0]);
    }

    #[test]
    fn itk_center_of_rotation_keeps_center_fixed() {
        // 180° about z, centered at (10, 0, 0) in ITK-LPS means: the LPS-x
        // through (10, 0, 0) is fixed. After LPS→RAS that maps the RAS
        // point (-10, 0, 0) to itself.
        let theta = std::f64::consts::PI;
        let mut m = Matrix3::identity();
        let (s, c) = (theta.sin(), theta.cos());
        m[(0, 0)] = c;
        m[(0, 1)] = -s;
        m[(1, 0)] = s;
        m[(1, 1)] = c;
        let a = Affine3::from_itk_components(m, Vector3::zeros(), Vector3::new(10.0, 0.0, 0.0));
        let p = a.apply([-10.0, 0.0, 0.0]);
        assert!((p[0] + 10.0).abs() < 1e-9);
        assert!(p[1].abs() < 1e-9);
        assert!(p[2].abs() < 1e-9);
    }
}
