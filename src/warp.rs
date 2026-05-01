//! Dense displacement-field component of a transform chain, in RAS+ mm.

use ndarray::Array4;

use crate::error::{Result, XfmError};
use crate::grid::TargetGrid;

/// Dense displacement field, with vectors and grid both in RAS+ mm.
///
/// `data` has shape `(nx, ny, nz, 3)`. At voxel `(i, j, k)` the displacement
/// added to a sampled world point is `(data[i, j, k, 0], _1, _2)`.
#[derive(Clone, Debug)]
pub struct DisplacementField {
    /// Displacement vectors, shape `(nx, ny, nz, 3)`, in RAS+ mm.
    pub data: Array4<f32>,
    /// Voxel-to-world grid the field is sampled on (RAS+ mm).
    pub grid: TargetGrid,
}

impl DisplacementField {
    /// Construct from displacement data and its grid.
    ///
    /// Returns [`XfmError::MalformedParameters`] if `data` is not
    /// 4-dimensional with a trailing axis of length 3, or if its leading
    /// dimensions disagree with `grid.dims`.
    pub fn new(data: Array4<f32>, grid: TargetGrid) -> Result<Self> {
        let shape = data.shape();
        if shape.len() != 4 || shape[3] != 3 {
            return Err(XfmError::MalformedParameters(format!(
                "displacement field must have shape (nx, ny, nz, 3), got {:?}",
                shape
            )));
        }
        let dims = [shape[0] as u64, shape[1] as u64, shape[2] as u64];
        if dims != grid.dims {
            return Err(XfmError::MalformedParameters(format!(
                "displacement field shape {:?} does not match grid dims {:?}",
                dims, grid.dims
            )));
        }
        Ok(Self { data, grid })
    }

    /// Trilinearly sample the displacement at a RAS+ world point.
    ///
    /// Out-of-bounds samples return a zero vector — the chain still applies
    /// the input point unchanged. This matches ITK's default extrapolation
    /// for `DisplacementFieldTransform`.
    pub fn sample(&self, world: [f64; 3]) -> [f64; 3] {
        let inv = match self.grid.inverse_affine() {
            Ok(m) => m,
            Err(_) => return [0.0; 3],
        };
        let v = inv * nalgebra::Vector4::new(world[0], world[1], world[2], 1.0);
        let (fx, fy, fz) = (v[0], v[1], v[2]);

        let (nx, ny, nz) = (
            self.data.shape()[0] as i64,
            self.data.shape()[1] as i64,
            self.data.shape()[2] as i64,
        );

        if !(fx.is_finite() && fy.is_finite() && fz.is_finite()) {
            return [0.0; 3];
        }
        // Hard out-of-bounds: outside [-0.5, n-0.5] in any axis → identity.
        if fx < -0.5 || fy < -0.5 || fz < -0.5 {
            return [0.0; 3];
        }
        if fx > nx as f64 - 0.5 || fy > ny as f64 - 0.5 || fz > nz as f64 - 0.5 {
            return [0.0; 3];
        }

        let i0 = fx.floor() as i64;
        let j0 = fy.floor() as i64;
        let k0 = fz.floor() as i64;
        let dx = fx - i0 as f64;
        let dy = fy - j0 as f64;
        let dz = fz - k0 as f64;

        let mut out = [0.0_f64; 3];
        for di in 0..2_i64 {
            for dj in 0..2_i64 {
                for dk in 0..2_i64 {
                    let i = (i0 + di).clamp(0, nx - 1) as usize;
                    let j = (j0 + dj).clamp(0, ny - 1) as usize;
                    let k = (k0 + dk).clamp(0, nz - 1) as usize;
                    let wx = if di == 0 { 1.0 - dx } else { dx };
                    let wy = if dj == 0 { 1.0 - dy } else { dy };
                    let wz = if dk == 0 { 1.0 - dz } else { dz };
                    let w = wx * wy * wz;
                    out[0] += w * self.data[(i, j, k, 0)] as f64;
                    out[1] += w * self.data[(i, j, k, 1)] as f64;
                    out[2] += w * self.data[(i, j, k, 2)] as f64;
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Matrix4;
    use ndarray::Array4;

    fn unit_grid(dims: [u64; 3]) -> TargetGrid {
        TargetGrid::from_matrix(Matrix4::identity(), dims)
    }

    #[test]
    fn zero_field_returns_zero() {
        let data = Array4::<f32>::zeros((4, 4, 4, 3));
        let f = DisplacementField::new(data, unit_grid([4, 4, 4])).unwrap();
        assert_eq!(f.sample([1.0, 1.0, 1.0]), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn constant_field_returns_constant() {
        let mut data = Array4::<f32>::zeros((4, 4, 4, 3));
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    data[(i, j, k, 0)] = 7.0;
                    data[(i, j, k, 1)] = -3.0;
                    data[(i, j, k, 2)] = 0.5;
                }
            }
        }
        let f = DisplacementField::new(data, unit_grid([4, 4, 4])).unwrap();
        let v = f.sample([1.5, 2.5, 0.5]);
        assert!((v[0] - 7.0).abs() < 1e-6);
        assert!((v[1] + 3.0).abs() < 1e-6);
        assert!((v[2] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn out_of_bounds_returns_zero() {
        let mut data = Array4::<f32>::zeros((4, 4, 4, 3));
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    data[(i, j, k, 0)] = 1.0;
                }
            }
        }
        let f = DisplacementField::new(data, unit_grid([4, 4, 4])).unwrap();
        // Way outside in +x.
        assert_eq!(f.sample([100.0, 0.0, 0.0]), [0.0, 0.0, 0.0]);
        // And in -x.
        assert_eq!(f.sample([-100.0, 0.0, 0.0]), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn linear_gradient_interpolates() {
        // dx field that ramps: dx(i, j, k) = i.
        let mut data = Array4::<f32>::zeros((4, 4, 4, 3));
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    data[(i, j, k, 0)] = i as f32;
                }
            }
        }
        let f = DisplacementField::new(data, unit_grid([4, 4, 4])).unwrap();
        // Voxel (1.5, 0, 0) should give dx = 1.5.
        let v = f.sample([1.5, 0.0, 0.0]);
        assert!((v[0] - 1.5).abs() < 1e-6, "got {:?}", v);
    }
}
