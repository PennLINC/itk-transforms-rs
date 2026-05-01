//! Reader for ITK Composite.h5 spatial transforms (ANTs `antsRegistration` output).
//!
//! Layout (from `ITK/Modules/IO/TransformHDF5/src/itkHDF5TransformIO.cxx`,
//! cross-checked against `nitransforms/io/itk.py`):
//!
//! ```text
//! /
//! ├── ITKVersion          (string)
//! ├── HDFVersion          (string)
//! ├── OSName              (string)
//! ├── OSVersion           (string)
//! └── TransformGroup/
//!     ├── 0/              # always a CompositeTransform wrapper — skipped
//!     │   └── TransformType
//!     ├── 1/              # actual component
//!     │   ├── TransformType
//!     │   ├── TransformParameters
//!     │   └── TransformFixedParameters
//!     ├── 2/
//!     │   ├── ...
//!     └── N/
//!         └── ...
//! ```

use std::path::Path;

use hdf5_metno::types::{FixedAscii, VarLenAscii, VarLenUnicode};
use hdf5_metno::{File, Group};
use nalgebra::{Matrix3, Vector3};
use ndarray::Array4;

use crate::affine::Affine3;
use crate::chain::TransformChain;
use crate::error::{Result, XfmError};
use crate::grid::TargetGrid;
use crate::lps_ras::affine_itk_to_ras;
use crate::warp::DisplacementField;

// ---- ITK layout constants ------------------------------------------------

/// Number of TransformParameters for a 3D affine: 3×3 matrix + 3-vector
/// translation.
const AFFINE_PARAM_COUNT: usize = 12;

/// Number of TransformFixedParameters for a 3D affine: a 3-vector center of
/// rotation.
const AFFINE_FIXED_PARAM_COUNT: usize = 3;

/// Number of TransformFixedParameters for a 3D `DisplacementFieldTransform`:
/// `[shape(3), origin(3), spacing(3), direction(9)]`.
const WARP_FIXED_PARAM_COUNT: usize = 18;

/// Vector dimension of a 3D displacement field (xyz components per voxel).
const WARP_VECTOR_DIM: usize = 3;

/// Maximum length we read for a fixed-length `TransformType` string. Real
/// ITK strings are well under 64 chars; 256 is generous.
const ITK_TRANSFORM_TYPE_MAX_LEN: usize = 256;

/// Read an ITK Composite.h5 transform file. Returns a [`TransformChain`]
/// already converted to RAS+ mm.
pub fn read_itk_h5(path: &Path) -> Result<TransformChain> {
    let file = File::open(path).map_err(|e| XfmError::InvalidFile {
        path: path.to_path_buf(),
        reason: format!("HDF5 open failed: {e}"),
    })?;

    let tg = file
        .group("TransformGroup")
        .map_err(|_| XfmError::InvalidFile {
            path: path.to_path_buf(),
            reason: "missing /TransformGroup root".to_string(),
        })?;

    let mut indexed = collect_transform_indices(&tg)?;
    indexed.sort_by_key(|(idx, _)| *idx);

    let mut chain = TransformChain::new();
    for (idx, name) in indexed {
        if idx == 0 {
            // The CompositeTransform wrapper. Skip — its only role is to
            // declare that this is a chain of the entries that follow.
            continue;
        }
        let g = tg.group(&name).map_err(|e| XfmError::InvalidFile {
            path: path.to_path_buf(),
            reason: format!("cannot open /TransformGroup/{name}: {e}"),
        })?;
        let ttype = read_transform_type(&g, path, &name)?;
        ingest_component(&mut chain, &g, &ttype, path)?;
    }

    if chain.is_empty() {
        return Err(XfmError::InvalidFile {
            path: path.to_path_buf(),
            reason: "no usable transform components found in /TransformGroup".to_string(),
        });
    }

    Ok(chain)
}

fn collect_transform_indices(tg: &Group) -> Result<Vec<(usize, String)>> {
    let names = tg.member_names()?;
    let mut out = Vec::new();
    for name in names {
        if let Ok(idx) = name.parse::<usize>() {
            out.push((idx, name));
        }
        // Non-numeric names (rare) are silently ignored.
    }
    Ok(out)
}

/// Try to read the `TransformType` dataset as a string. ITK stores this as
/// a 1-element 1D array (shape `(1,)`) of either variable-length or
/// fixed-length strings, depending on the writer's version. We try scalars
/// and 1D arrays of various string flavours and return the first hit.
fn read_transform_type(g: &Group, path: &Path, group_name: &str) -> Result<String> {
    let ds = g
        .dataset("TransformType")
        .map_err(|_| XfmError::InvalidFile {
            path: path.to_path_buf(),
            reason: format!("/TransformGroup/{group_name}/TransformType missing"),
        })?;

    // Variable-length strings (most common in modern ITK output).
    if let Ok(v) = ds.read_raw::<VarLenAscii>() {
        if let Some(first) = v.into_iter().next() {
            return Ok(first.to_string());
        }
    }
    if let Ok(v) = ds.read_raw::<VarLenUnicode>() {
        if let Some(first) = v.into_iter().next() {
            return Ok(first.to_string());
        }
    }
    if let Ok(v) = ds.read_scalar::<VarLenUnicode>() {
        return Ok(v.to_string());
    }
    if let Ok(v) = ds.read_scalar::<VarLenAscii>() {
        return Ok(v.to_string());
    }
    // Fixed-length fallbacks. The longest ITK TransformType string we expect
    // is around 40 chars; 256 is generous.
    if let Ok(v) = ds.read_raw::<FixedAscii<ITK_TRANSFORM_TYPE_MAX_LEN>>() {
        if let Some(first) = v.into_iter().next() {
            return Ok(first.as_str().to_string());
        }
    }
    if let Ok(v) = ds.read_scalar::<FixedAscii<ITK_TRANSFORM_TYPE_MAX_LEN>>() {
        return Ok(v.as_str().to_string());
    }

    Err(XfmError::InvalidFile {
        path: path.to_path_buf(),
        reason: format!("could not decode /TransformGroup/{group_name}/TransformType as a string"),
    })
}

fn ingest_component(chain: &mut TransformChain, g: &Group, ttype: &str, path: &Path) -> Result<()> {
    if ttype.starts_with("CompositeTransform") {
        // Sometimes a non-zero index is also a wrapper. Skip.
        return Ok(());
    }
    if ttype.starts_with("AffineTransform")
        || ttype.starts_with("MatrixOffsetTransformBase")
        || ttype.starts_with("ScaleSkewVersor3DTransform")
        || ttype.starts_with("Rigid3DTransform")
        || ttype.starts_with("Similarity3DTransform")
        || ttype.starts_with("Euler3DTransform")
        || ttype.starts_with("VersorRigid3DTransform")
        || ttype.starts_with("ScaleVersor3DTransform")
        || ttype.starts_with("FixedCenterOfRotationAffineTransform")
    {
        chain.push_affine(read_affine(g, path)?);
        return Ok(());
    }
    if ttype.starts_with("DisplacementFieldTransform") {
        chain.push_warp(read_warp(g, path)?)?;
        return Ok(());
    }
    Err(XfmError::UnsupportedTransformType(ttype.to_string()))
}

/// Read an `AffineTransform_*_3_3` (or compatible MatrixOffset-derived
/// transform). Components: 3×3 matrix, 3-vector translation, 3-vector center
/// of rotation (fixed parameters). Output: [`Affine3`] in RAS+ mm.
fn read_affine(g: &Group, path: &Path) -> Result<Affine3> {
    let params: Vec<f64> = read_doubles(g, "TransformParameters", path)?;
    let fixed: Vec<f64> = read_doubles(g, "TransformFixedParameters", path)?;

    if params.len() != AFFINE_PARAM_COUNT {
        return Err(XfmError::MalformedParameters(format!(
            "AffineTransform expects {AFFINE_PARAM_COUNT} TransformParameters, got {}",
            params.len()
        )));
    }
    if fixed.len() != AFFINE_FIXED_PARAM_COUNT {
        return Err(XfmError::MalformedParameters(format!(
            "AffineTransform expects {AFFINE_FIXED_PARAM_COUNT} TransformFixedParameters (center), got {}",
            fixed.len()
        )));
    }

    let matrix = Matrix3::<f64>::new(
        params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7],
        params[8],
    );
    let translation = Vector3::new(params[9], params[10], params[11]);
    let center = Vector3::new(fixed[0], fixed[1], fixed[2]);

    Ok(Affine3::from_itk_components(matrix, translation, center))
}

/// Read a `DisplacementFieldTransform_*_3_3` and return a [`DisplacementField`]
/// in RAS+ mm. Layout per nitransforms / ITK source:
///
/// - FixedParameters: `[shape(3), origin(3), spacing(3), direction(9)]`
/// - Parameters: `3 * nx * ny * nz` floats, Fortran-order with leading 3 as
///   the vector dimension. We unflatten manually so the resulting ndarray is
///   `(nx, ny, nz, 3)` in C order.
/// - Vectors are LPS+ — flip `x` and `y` components.
/// - Grid affine `A_itk = from_matvec(direction · diag(spacing), origin)`,
///   converted to RAS+ via the LPS sandwich.
fn read_warp(g: &Group, path: &Path) -> Result<DisplacementField> {
    let fixed: Vec<f64> = read_doubles(g, "TransformFixedParameters", path)?;
    if fixed.len() != WARP_FIXED_PARAM_COUNT {
        return Err(XfmError::MalformedParameters(format!(
            "DisplacementFieldTransform expects {WARP_FIXED_PARAM_COUNT} TransformFixedParameters, got {}",
            fixed.len()
        )));
    }

    let nx = fixed[0].round() as usize;
    let ny = fixed[1].round() as usize;
    let nz = fixed[2].round() as usize;
    if nx == 0 || ny == 0 || nz == 0 {
        return Err(XfmError::MalformedParameters(format!(
            "DisplacementFieldTransform has zero dimension: ({nx}, {ny}, {nz})"
        )));
    }

    let origin = Vector3::new(fixed[3], fixed[4], fixed[5]);
    let spacing = [fixed[6], fixed[7], fixed[8]];
    let direction = Matrix3::<f64>::new(
        fixed[9], fixed[10], fixed[11], fixed[12], fixed[13], fixed[14], fixed[15], fixed[16],
        fixed[17],
    );

    // Build ITK grid affine: rotation*scale block from (direction · diag(spacing)),
    // translation = origin.
    let mut linear = direction;
    for col in 0..3 {
        let s = spacing[col];
        for row in 0..3 {
            linear[(row, col)] *= s;
        }
    }
    let mut itk_affine = nalgebra::Matrix4::identity();
    itk_affine.fixed_view_mut::<3, 3>(0, 0).copy_from(&linear);
    itk_affine[(0, 3)] = origin[0];
    itk_affine[(1, 3)] = origin[1];
    itk_affine[(2, 3)] = origin[2];

    let ras_affine = affine_itk_to_ras(&itk_affine);
    let grid = TargetGrid::from_matrix(ras_affine, [nx as u64, ny as u64, nz as u64]);

    let total = WARP_VECTOR_DIM
        .checked_mul(nx)
        .and_then(|v| v.checked_mul(ny))
        .and_then(|v| v.checked_mul(nz));
    let total = total.ok_or_else(|| {
        XfmError::MalformedParameters("displacement field too large for usize".into())
    })?;

    let raw: Vec<f32> = read_floats(g, "TransformParameters", path)?;
    if raw.len() != total {
        return Err(XfmError::MalformedParameters(format!(
            "DisplacementFieldTransform expects {} TransformParameters values for shape \
             ({nx}, {ny}, {nz}, 3), got {}",
            total,
            raw.len(),
        )));
    }

    // Fortran-order with leading vec_dim=3 means: flat[c + 3*(i + nx*(j + ny*k))]
    // = field(c, i, j, k). Build an (nx, ny, nz, 3) C-order ndarray, flipping
    // the LPS x and y components on the way through.
    let mut data = Array4::<f32>::zeros((nx, ny, nz, WARP_VECTOR_DIM));
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let base = WARP_VECTOR_DIM * (i + nx * (j + ny * k));
                let vx = raw[base];
                let vy = raw[base + 1];
                let vz = raw[base + 2];
                // ITK stores LPS displacement vectors → flip x,y for RAS+.
                data[(i, j, k, 0)] = -vx;
                data[(i, j, k, 1)] = -vy;
                data[(i, j, k, 2)] = vz;
            }
        }
    }

    DisplacementField::new(data, grid)
}

fn read_doubles(g: &Group, name: &str, path: &Path) -> Result<Vec<f64>> {
    let ds = g.dataset(name).map_err(|_| XfmError::InvalidFile {
        path: path.to_path_buf(),
        reason: format!("missing dataset '{name}' in transform group"),
    })?;
    Ok(ds.read_raw::<f64>()?)
}

fn read_floats(g: &Group, name: &str, path: &Path) -> Result<Vec<f32>> {
    let ds = g.dataset(name).map_err(|_| XfmError::InvalidFile {
        path: path.to_path_buf(),
        reason: format!("missing dataset '{name}' in transform group"),
    })?;
    Ok(ds.read_raw::<f32>()?)
}
