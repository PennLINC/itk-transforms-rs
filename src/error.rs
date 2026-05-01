//! Error types for transform reading and composition.

use std::path::PathBuf;

use thiserror::Error;

/// Anything that can go wrong while reading or composing an ITK transform.
#[derive(Debug, Error)]
pub enum XfmError {
    /// I/O failure (file not found, permission denied, partial read, etc.).
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// HDF5 access failure surfaced from `hdf5-metno`.
    #[error("HDF5 error: {0}")]
    Hdf5(#[from] hdf5_metno::Error),

    /// NIfTI access failure surfaced from `nifti`.
    #[error("NIfTI error: {0}")]
    Nifti(#[from] nifti::NiftiError),

    /// File exists and is readable but does not match an ITK transform layout.
    #[error("file {path} is not a recognised ITK Composite.h5 transform: {reason}")]
    InvalidFile {
        /// Path of the offending file.
        path: PathBuf,
        /// Human-readable explanation of what was wrong.
        reason: String,
    },

    /// File parses but contains a transform type the reader does not handle
    /// (e.g. `BSplineTransform`, `QuaternionRigidTransform`).
    #[error("unsupported ITK transform type: {0}")]
    UnsupportedTransformType(String),

    /// Parameter dataset is the wrong shape, length, or contains nonsense.
    #[error("malformed transform parameters: {0}")]
    MalformedParameters(String),

    /// Caller asked to invert a chain that contains a displacement field
    /// (numerical warp inversion is out of scope).
    #[error("transform component is not invertible: {0}")]
    NotInvertible(String),

    /// 4×4 affine has no inverse.
    #[error("singular matrix")]
    SingularMatrix,

    /// A multi-warp chain contains displacement fields whose grids
    /// (dimensions or voxel-to-world affine) disagree. The first warp's
    /// grid would be silently retained as `default_target_grid`, so this
    /// is rejected explicitly.
    #[error("incompatible displacement-field grids in chain: {0}")]
    IncompatibleGrid(String),
}

/// Convenience alias for `Result<T, XfmError>`.
pub type Result<T> = std::result::Result<T, XfmError>;
