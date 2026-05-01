//! Target output grid: voxel-to-world affine plus 3D dimensions, in RAS+ mm.

use std::path::Path;

use nalgebra::Matrix4;
use nifti::{NiftiObject, ReaderOptions};

use crate::error::{Result, XfmError};

/// A regular 3D voxel grid in RAS+ mm.
///
/// # Examples
///
/// ```
/// use itk_transforms_rs::TargetGrid;
/// use nalgebra::Matrix4;
///
/// let mut a = Matrix4::<f64>::identity();
/// a[(0, 0)] = 2.0;          // 2 mm voxel along i
/// a[(1, 1)] = 2.0;
/// a[(2, 2)] = 2.0;
/// let g = TargetGrid::from_matrix(a, [128, 128, 64]);
/// assert_eq!(g.voxel_sizes(), [2.0, 2.0, 2.0]);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct TargetGrid {
    /// Row-major 4×4 voxel-to-world affine in RAS+ mm. `[i, j, k, 1]` → world.
    pub affine: [[f64; 4]; 4],
    /// `(nx, ny, nz)`.
    pub dims: [u64; 3],
}

impl TargetGrid {
    /// Create from explicit affine + dims.
    pub fn new(affine: [[f64; 4]; 4], dims: [u64; 3]) -> Self {
        Self { affine, dims }
    }

    /// Create from a nalgebra matrix + dims.
    pub fn from_matrix(affine: Matrix4<f64>, dims: [u64; 3]) -> Self {
        let mut a = [[0.0_f64; 4]; 4];
        for r in 0..4 {
            for c in 0..4 {
                a[r][c] = affine[(r, c)];
            }
        }
        Self { affine: a, dims }
    }

    /// Return the affine as a nalgebra matrix.
    pub fn affine_matrix(&self) -> Matrix4<f64> {
        Matrix4::from_row_slice(&[
            self.affine[0][0],
            self.affine[0][1],
            self.affine[0][2],
            self.affine[0][3],
            self.affine[1][0],
            self.affine[1][1],
            self.affine[1][2],
            self.affine[1][3],
            self.affine[2][0],
            self.affine[2][1],
            self.affine[2][2],
            self.affine[2][3],
            self.affine[3][0],
            self.affine[3][1],
            self.affine[3][2],
            self.affine[3][3],
        ])
    }

    /// Return the inverse affine (world-to-voxel).
    pub fn inverse_affine(&self) -> Result<Matrix4<f64>> {
        self.affine_matrix()
            .try_inverse()
            .ok_or(XfmError::SingularMatrix)
    }

    /// Voxel spacings — Euclidean lengths of each column of the linear part.
    pub fn voxel_sizes(&self) -> [f64; 3] {
        let m = self.affine_matrix();
        [
            (m[(0, 0)].powi(2) + m[(1, 0)].powi(2) + m[(2, 0)].powi(2)).sqrt(),
            (m[(0, 1)].powi(2) + m[(1, 1)].powi(2) + m[(2, 1)].powi(2)).sqrt(),
            (m[(0, 2)].powi(2) + m[(1, 2)].powi(2) + m[(2, 2)].powi(2)).sqrt(),
        ]
    }

    /// Load grid metadata from a NIfTI file (uses sform/qform per the standard;
    /// returns whatever the `nifti` crate composes — already RAS+ for any
    /// well-formed NIfTI).
    pub fn from_nifti<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let obj = ReaderOptions::new().read_file(path)?;
        let header = obj.header();
        let affine = header.affine::<f64>();

        let dims = [
            u64::from(header.dim[1]),
            u64::from(header.dim[2]),
            u64::from(header.dim[3]),
        ];

        if dims[0] == 0 || dims[1] == 0 || dims[2] == 0 {
            return Err(XfmError::InvalidFile {
                path: path.to_path_buf(),
                reason: format!("NIfTI has zero spatial dimensions: {:?}", dims),
            });
        }

        Ok(Self::from_matrix(affine, dims))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voxel_sizes_iso() {
        let mut a = Matrix4::identity();
        a[(0, 0)] = 2.0;
        a[(1, 1)] = 2.0;
        a[(2, 2)] = 2.0;
        let g = TargetGrid::from_matrix(a, [10, 10, 10]);
        assert_eq!(g.voxel_sizes(), [2.0, 2.0, 2.0]);
    }

    #[test]
    fn inverse_affine_round_trip() {
        let mut a = Matrix4::identity();
        a[(0, 3)] = 5.0;
        a[(0, 0)] = 2.0;
        let g = TargetGrid::from_matrix(a, [10, 10, 10]);
        let inv = g.inverse_affine().unwrap();
        let p = inv * a * nalgebra::Vector4::new(1.0, 2.0, 3.0, 1.0);
        // Should round-trip to (1, 2, 3, 1).
        assert!((p[0] - 1.0).abs() < 1e-12);
        assert!((p[1] - 2.0).abs() < 1e-12);
        assert!((p[2] - 3.0).abs() < 1e-12);
    }
}
