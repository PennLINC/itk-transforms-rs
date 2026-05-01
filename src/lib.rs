//! ITK composite spatial transforms in pure Rust, normalised to RAS+.
//!
//! This crate parses the `*Composite.h5` files written by ANTs
//! `antsRegistration` and the `Insight Transform File V1.0` plain-text format
//! used by ITK and ANTs for affines. Components are returned as a
//! [`TransformChain`] of [`Affine3`] and [`DisplacementField`] entries.
//!
//! # Coordinate convention
//!
//! ITK works in **LPS+** (Left-Posterior-Superior). NIfTI / ODX / TRX /
//! MRtrix work in **RAS+** (Right-Anterior-Superior) mm. Everything this
//! crate hands back to the caller — affines, warp grids, and warp vectors —
//! is in RAS+ mm. The single LPS↔RAS boundary lives in an internal module
//! (`lps_ras`) and is validated against MRtrix3 and nitransforms; callers
//! never need to apply their own sign-flips.
//!
//! # Composition direction
//!
//! Chains are applied in stored order:
//! `chain(p) = c_N(c_{N−1}( … c_1(p)))`. For ANTs Composite.h5 this maps
//! **fixed-image space → moving-image space**, matching the pull-based
//! semantics of `antsApplyTransforms`.
//!
//! # Quick start
//!
//! ```no_run
//! use itk_transforms_rs::{read_itk, Result};
//! use std::path::Path;
//!
//! fn main() -> Result<()> {
//!     let chain = read_itk(Path::new("Composite.h5"))?;
//!     let p_moved = chain.map_point([10.0, 20.0, 30.0]);
//!     println!("{:?}", p_moved);
//!     Ok(())
//! }
//! ```

#![warn(missing_debug_implementations)]
#![warn(rust_2018_idioms)]

pub mod affine;
pub mod chain;
pub mod error;
pub mod grid;
pub mod warp;

pub(crate) mod itk_h5;
pub(crate) mod itk_txt;
pub(crate) mod lps_ras;

pub use crate::affine::Affine3;
pub use crate::chain::{TransformChain, TransformComponent};
pub use crate::error::{Result, XfmError};
pub use crate::grid::TargetGrid;
pub use crate::itk_h5::read_itk_h5;
pub use crate::itk_txt::read_itk_txt;
pub use crate::warp::DisplacementField;

use std::path::Path;

/// Read an ITK transform from a file, dispatching on extension.
///
/// - `.h5` → [`read_itk_h5`] (composite affine + warp)
/// - `.txt` → [`read_itk_txt`] (Insight Transform File V1.0, affine-only)
///
/// Anything else returns [`XfmError::InvalidFile`].
pub fn read_itk(path: &Path) -> Result<TransformChain> {
    match path.extension().and_then(|e| e.to_str()) {
        Some(ext) if ext.eq_ignore_ascii_case("h5") => read_itk_h5(path),
        Some(ext) if ext.eq_ignore_ascii_case("txt") => read_itk_txt(path),
        _ => Err(XfmError::InvalidFile {
            path: path.to_path_buf(),
            reason: "expected .h5 (Composite) or .txt (Insight Transform File)".into(),
        }),
    }
}
