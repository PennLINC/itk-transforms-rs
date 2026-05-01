//! Composed transform: a list of affine and/or warp components, in RAS+ mm.
//!
//! Applied in stored order: `chain(p) = c_N(c_{N-1}(... c_1(p)))`. For ANTs
//! Composite.h5 the chain maps **fixed-space coordinates → moving-space
//! coordinates** (the convention `antsApplyTransforms` uses for pull-based
//! resampling of moving → fixed). Callers wanting the opposite direction can
//! call [`TransformChain::invert`] (only supported when no warp components
//! are present, since numerical warp inversion is out of scope).

use nalgebra::Matrix3;

use crate::affine::Affine3;
use crate::error::{Result, XfmError};
use crate::grid::TargetGrid;
use crate::warp::DisplacementField;

/// One component of a [`TransformChain`].
#[derive(Clone, Debug)]
pub enum TransformComponent {
    /// Linear affine transform (rotation, scale, shear, translation) in RAS+ mm.
    Affine(Affine3),
    /// Dense displacement field; the sampled vector is added to the input point.
    Warp(DisplacementField),
}

impl TransformComponent {
    /// Apply this component to a point in RAS+ mm.
    pub fn apply(&self, p: [f64; 3]) -> [f64; 3] {
        match self {
            Self::Affine(a) => a.apply(p),
            Self::Warp(w) => {
                let d = w.sample(p);
                [p[0] + d[0], p[1] + d[1], p[2] + d[2]]
            }
        }
    }

    /// 3×3 Jacobian of this component at point `p`. Affines return their
    /// linear part exactly; warps approximate `I + ∂displacement/∂p` via
    /// central finite differences using `step` (in world mm).
    pub fn jacobian_at(&self, p: [f64; 3], step: f64) -> Matrix3<f64> {
        match self {
            Self::Affine(a) => a.linear(),
            Self::Warp(w) => {
                let mut j = Matrix3::<f64>::identity();
                for axis in 0..3 {
                    let mut p_plus = p;
                    let mut p_minus = p;
                    p_plus[axis] += step;
                    p_minus[axis] -= step;
                    let d_plus = w.sample(p_plus);
                    let d_minus = w.sample(p_minus);
                    let inv2 = 1.0 / (2.0 * step);
                    for row in 0..3 {
                        j[(row, axis)] += (d_plus[row] - d_minus[row]) * inv2;
                    }
                }
                j
            }
        }
    }

    /// `true` if this is the [`TransformComponent::Affine`] variant.
    pub fn is_affine(&self) -> bool {
        matches!(self, Self::Affine(_))
    }

    /// `true` if this is the [`TransformComponent::Warp`] variant.
    pub fn is_warp(&self) -> bool {
        matches!(self, Self::Warp(_))
    }
}

/// A composed spatial transform.
///
/// Components are stored in application order; see the module docs for
/// composition semantics.
#[derive(Clone, Debug, Default)]
pub struct TransformChain {
    /// The components, applied left-to-right.
    pub components: Vec<TransformComponent>,
    /// Optional default output grid, populated when an ITK h5 contains a
    /// displacement field (its grid is the conventional fixed-image grid).
    /// All warps in the chain must share this grid; mismatches are rejected
    /// by [`Self::push_warp`].
    pub default_target_grid: Option<TargetGrid>,
}

impl TransformChain {
    /// Create an empty chain (acts as the identity).
    pub fn new() -> Self {
        Self::default()
    }

    /// Append an affine component.
    pub fn push_affine(&mut self, a: Affine3) {
        self.components.push(TransformComponent::Affine(a));
    }

    /// Append a displacement-field component.
    ///
    /// The first warp's grid is captured as [`Self::default_target_grid`].
    /// Subsequent warps must share that grid (same `dims` and a
    /// voxel-to-world affine equal to within `1e-9` mm); otherwise the
    /// silent loss of grid metadata is rejected with
    /// [`XfmError::IncompatibleGrid`].
    pub fn push_warp(&mut self, w: DisplacementField) -> Result<()> {
        match &self.default_target_grid {
            None => {
                self.default_target_grid = Some(w.grid.clone());
            }
            Some(existing) => {
                if !grids_match(existing, &w.grid) {
                    return Err(XfmError::IncompatibleGrid(format!(
                        "new warp grid (dims {:?}) disagrees with existing chain grid (dims {:?})",
                        w.grid.dims, existing.dims,
                    )));
                }
            }
        }
        self.components.push(TransformComponent::Warp(w));
        Ok(())
    }

    /// `true` if no components have been pushed.
    pub fn is_empty(&self) -> bool {
        self.components.is_empty()
    }

    /// `true` if any component is a displacement field.
    pub fn has_warp(&self) -> bool {
        self.components.iter().any(|c| c.is_warp())
    }

    /// Apply the chain to a point.
    ///
    /// # Examples
    ///
    /// ```
    /// use itk_transforms_rs::{Affine3, TransformChain};
    /// use nalgebra::Matrix4;
    ///
    /// let mut a = Matrix4::<f64>::identity();
    /// a[(0, 3)] = 5.0;                  // RAS+ translation along x
    /// let mut chain = TransformChain::new();
    /// chain.push_affine(Affine3::from_matrix(a));
    /// assert_eq!(chain.map_point([0.0, 0.0, 0.0]), [5.0, 0.0, 0.0]);
    /// ```
    pub fn map_point(&self, mut p: [f64; 3]) -> [f64; 3] {
        for c in &self.components {
            p = c.apply(p);
        }
        p
    }

    /// Compose Jacobians via chain rule: J_total = J_N · ... · J_2 · J_1.
    /// `step` is the FD step (in world mm) used for warp components; pass a
    /// value comparable to the target grid's voxel size (~min spacing).
    pub fn jacobian_at(&self, p: [f64; 3], step: f64) -> Matrix3<f64> {
        let mut p_curr = p;
        let mut j_total = Matrix3::<f64>::identity();
        for c in &self.components {
            let j_i = c.jacobian_at(p_curr, step);
            j_total = j_i * j_total;
            p_curr = c.apply(p_curr);
        }
        j_total
    }

    /// Reverse the chain. Affines invert exactly; any warp component triggers
    /// `XfmError::NotInvertible` (numerical warp inversion is v2 work).
    pub fn invert(self) -> Result<Self> {
        if self.has_warp() {
            return Err(XfmError::NotInvertible(
                "displacement-field component cannot be numerically inverted in v1".to_string(),
            ));
        }
        let mut inverted = Vec::with_capacity(self.components.len());
        for c in self.components.into_iter().rev() {
            match c {
                TransformComponent::Affine(a) => {
                    inverted.push(TransformComponent::Affine(a.try_inverse()?))
                }
                TransformComponent::Warp(_) => unreachable!("checked above"),
            }
        }
        Ok(Self {
            components: inverted,
            default_target_grid: None,
        })
    }
}

/// Two grids are considered identical if their dims match exactly and their
/// 4×4 affines agree element-wise within `1e-9` mm. Displacement fields are
/// authored on a single fixed grid in practice, so anything looser than
/// floating-point noise is a real mismatch.
fn grids_match(a: &TargetGrid, b: &TargetGrid) -> bool {
    if a.dims != b.dims {
        return false;
    }
    for r in 0..4 {
        for c in 0..4 {
            if (a.affine[r][c] - b.affine[r][c]).abs() > 1e-9 {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Matrix4;

    fn translation_affine(t: [f64; 3]) -> Affine3 {
        let mut m = Matrix4::identity();
        m[(0, 3)] = t[0];
        m[(1, 3)] = t[1];
        m[(2, 3)] = t[2];
        Affine3::from_matrix(m)
    }

    #[test]
    fn empty_chain_is_identity() {
        let c = TransformChain::new();
        assert_eq!(c.map_point([1.0, 2.0, 3.0]), [1.0, 2.0, 3.0]);
        assert_eq!(c.jacobian_at([0.0; 3], 1.0), Matrix3::identity());
    }

    #[test]
    fn affine_chain_composes() {
        let mut c = TransformChain::new();
        c.push_affine(translation_affine([1.0, 0.0, 0.0]));
        c.push_affine(translation_affine([0.0, 2.0, 0.0]));
        c.push_affine(translation_affine([0.0, 0.0, 3.0]));
        assert_eq!(c.map_point([0.0; 3]), [1.0, 2.0, 3.0]);
    }

    #[test]
    fn invert_translation() {
        let mut c = TransformChain::new();
        c.push_affine(translation_affine([5.0, -2.0, 1.0]));
        let inv = c.invert().unwrap();
        let p = inv.map_point([5.0, -2.0, 1.0]);
        assert!(p[0].abs() < 1e-12);
        assert!(p[1].abs() < 1e-12);
        assert!(p[2].abs() < 1e-12);
    }

    #[test]
    fn cannot_invert_warp() {
        use ndarray::Array4;
        let mut c = TransformChain::new();
        let data = Array4::<f32>::zeros((2, 2, 2, 3));
        let grid = TargetGrid::from_matrix(Matrix4::identity(), [2, 2, 2]);
        c.push_warp(DisplacementField::new(data, grid).unwrap())
            .unwrap();
        assert!(matches!(c.invert(), Err(XfmError::NotInvertible(_))));
    }

    #[test]
    fn second_warp_with_mismatched_grid_is_rejected() {
        use ndarray::Array4;
        let mut c = TransformChain::new();
        let data = Array4::<f32>::zeros((2, 2, 2, 3));
        let grid = TargetGrid::from_matrix(Matrix4::identity(), [2, 2, 2]);
        c.push_warp(DisplacementField::new(data, grid).unwrap())
            .unwrap();

        // Same dims but a shifted affine.
        let mut other_aff = Matrix4::<f64>::identity();
        other_aff[(0, 3)] = 5.0;
        let other = DisplacementField::new(
            Array4::<f32>::zeros((2, 2, 2, 3)),
            TargetGrid::from_matrix(other_aff, [2, 2, 2]),
        )
        .unwrap();
        assert!(matches!(
            c.push_warp(other),
            Err(XfmError::IncompatibleGrid(_))
        ));

        // And different dims.
        let other_dims = DisplacementField::new(
            Array4::<f32>::zeros((4, 4, 4, 3)),
            TargetGrid::from_matrix(Matrix4::identity(), [4, 4, 4]),
        )
        .unwrap();
        assert!(matches!(
            c.push_warp(other_dims),
            Err(XfmError::IncompatibleGrid(_))
        ));
    }

    #[test]
    fn jacobian_of_zero_warp_is_identity() {
        use ndarray::Array4;
        let mut c = TransformChain::new();
        let data = Array4::<f32>::zeros((4, 4, 4, 3));
        let grid = TargetGrid::from_matrix(Matrix4::identity(), [4, 4, 4]);
        c.push_warp(DisplacementField::new(data, grid).unwrap())
            .unwrap();
        let j = c.jacobian_at([1.5, 1.5, 1.5], 0.5);
        for row in 0..3 {
            for col in 0..3 {
                let want = if row == col { 1.0 } else { 0.0 };
                assert!((j[(row, col)] - want).abs() < 1e-9);
            }
        }
    }
}
