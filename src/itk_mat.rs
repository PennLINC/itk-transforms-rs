//! Reader for ITK transform files in **MATLAB Level 4 MAT-file** format
//! (`.mat`).
//!
//! These files are bit-for-bit valid [MATLAB v4 MAT-files][mat-spec] and can
//! be opened by MATLAB itself, GNU Octave, `scipy.io.loadmat`, or any other
//! MAT-v4 consumer. ANTs `antsRegistration` writes them as
//! `*0GenericAffine.mat` via the standard chain
//! `itk::TransformFileWriterTemplate` → `MatlabTransformIO::Write` → VNL's
//! `vnl_matlab_write` (see ITK's
//! [`itkMatlabTransformIO.cxx`][itk-mat] and VXL's
//! [`vnl_matlab_write.cxx`][vnl-write]).
//!
//! What ITK adds on top of the format is a **convention** — not a new file
//! type. Each variable in the file is a plain MATLAB column vector of
//! doubles; ITK uses the variable *name* as the transform-class tag and
//! interprets the numeric payload as ITK's `Parameters` / `FixedParameters`
//! arrays.
//!
//! [mat-spec]: https://data.cresis.ku.edu/data/mat_reader_files/matlab_4_mat_file_format.pdf
//! [itk-mat]: https://github.com/InsightSoftwareConsortium/ITK/blob/master/Modules/IO/TransformMatlab/src/itkMatlabTransformIO.cxx
//! [vnl-write]: https://github.com/vxl/vxl/blob/master/core/vnl/vnl_matlab_write.cxx
//!
//! # Per-transform record
//!
//! For each component in `MatlabTransformIO::Write`'s loop, ITK emits two
//! consecutive MAT-file variables:
//!
//! 1. **Parameters** — variable name is the transform class (e.g.
//!    `AffineTransform_double_3_3`); data is `Transform::GetParameters()`
//!    as an `N × 1` column. For `MatrixOffsetTransformBase`-derived
//!    transforms (Affine, Similarity, Rigid3D, ScaleSkew/ScaleVersor) `N
//!    = 12`: 9 row-major matrix elements followed by 3 translation
//!    components. Other transform types pack their parameters
//!    differently; this reader currently accepts only the 12-parameter
//!    layout.
//! 2. **Fixed parameters** — variable name is the literal string `fixed`
//!    (per `vnl_matlab_write(out, ..., "fixed")`); data is
//!    `Transform::GetFixedParameters()`. For 3D affines that's 3 doubles
//!    (centre of rotation).
//!
//! The semantics match the equivalent `TransformParameters` /
//! `TransformFixedParameters` datasets inside an h5 composite, so we reuse
//! [`Affine3::from_itk_components`] for the LPS→RAS sandwich and
//! centre-of-rotation handling.
//!
//! # Format scope (what `.mat` does and does not carry)
//!
//! ITK's `MatlabTransformIO::CanWriteFile` accepts any `.mat` extension and
//! does not special-case transform types — in principle a displacement
//! field could be written here too. In practice, ANTs (and the wider ITK
//! ecosystem) only writes `.mat` for affine-class transforms, sending
//! displacement fields to the HDF5 / NIfTI / `.nii.gz` paths in
//! [`itkantsReadWriteTransform.h`][ants-rw]. To stay safe, this reader
//! rejects transform-class names it does not recognise as affine via
//! [`XfmError::UnsupportedTransformType`]; warps live in `.h5`, not `.mat`.
//!
//! [ants-rw]: https://github.com/ANTsX/ANTs/blob/master/Utilities/itkantsReadWriteTransform.h
//!
//! # MATLAB Level 4 variable header
//!
//! Each variable starts with a 20-byte header — five `int32`s, in the
//! file's byte order — followed by the variable name (length and trailing
//! NUL given by `namlen`) and then `rows × cols` numeric values.
//!
//! | offset | bytes | field   | meaning                                          |
//! |--------|-------|---------|--------------------------------------------------|
//! | 0      | 4     | type    | precision + storage + byte-order flags (below)   |
//! | 4      | 4     | rows    | M                                                |
//! | 8      | 4     | cols    | N                                                |
//! | 12     | 4     | imag    | 1 if complex, else 0 (always 0 for transforms)   |
//! | 16     | 4     | namlen  | length of name **including** trailing NUL        |
//!
//! ## The `type` field, per VNL's encoding
//!
//! The MAT-v4 spec splits `type` into four decimal digits as `M·O·P·T`
//! (machine, reserved, precision, matrix-type). VXL/VNL repurposes the
//! first two digits — see
//! [`vnl_matlab_header.h`][vnl-header]:
//!
//! ```text
//! type = (1000 · is_big_endian) + (100 · is_row_wise) + (10 · is_single)
//! ```
//!
//! Concretely:
//! - thousands digit (`M` slot): byte order — 0 = little-endian, 1 = big-endian
//! - hundreds digit (`O` slot, formally reserved): 0 = column-wise, 1 = row-wise
//! - tens digit (`P` slot): 0 = double, 1 = single precision
//! - units digit (`T` slot): VNL writers always emit 0 (numeric full)
//!
//! ITK's writer composes this as
//! `type = native_BYTE_ORDER + vnl_COLUMN_WISE + vnl_scalar_precision(x)`,
//! so on a little-endian host with double parameters the on-disk `type` is
//! exactly 0 — which is what `*0GenericAffine.mat` files contain.
//!
//! [vnl-header]: https://github.com/vxl/vxl/blob/master/core/vnl/vnl_matlab_header.h
//!
//! ## Byte-order auto-detection
//!
//! Per [`vnl_matlab_read.cxx`'s logic][vnl-read], a `type` value parsed
//! little-endian as one of `{0, 10, 100, 110, 1000, 1100, 1110}` is
//! considered "native byte order"; any other value means the header was
//! written on the opposite-endian host and every header field needs
//! swapping. We mirror that here: read `type` as little-endian, look it up
//! in the allow-list, and swap all five header `int32`s if the lookup
//! misses. Numeric payloads are then decoded with the same swap policy.
//!
//! [vnl-read]: https://github.com/vxl/vxl/blob/master/core/vnl/vnl_matlab_read.cxx

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use nalgebra::{Matrix3, Vector3};

use crate::affine::Affine3;
use crate::chain::TransformChain;
use crate::error::{Result, XfmError};

const HEADER_BYTES: usize = 20;
const AFFINE_PARAM_COUNT: usize = 12;
const AFFINE_FIXED_PARAM_COUNT: usize = 3;

/// MATLAB v4 fixed header (20 bytes), decoded.
#[derive(Debug)]
struct RawHeader {
    rows: usize,
    cols: usize,
    namlen: usize,
    is_single_precision: bool,
    needs_swap: bool,
}

/// One full MATLAB v4 variable: header + name + decoded numeric payload.
#[derive(Debug)]
struct Variable {
    name: String,
    values: Vec<f64>,
}

/// Read a MATLAB v4 ITK transform file (`.mat`) and return a
/// [`TransformChain`] in RAS+ mm.
///
/// `.mat` files contain affine components only; encountering anything else
/// returns [`XfmError::UnsupportedTransformType`].
pub fn read_itk_mat(path: &Path) -> Result<TransformChain> {
    let file = File::open(path).map_err(|e| XfmError::InvalidFile {
        path: path.to_path_buf(),
        reason: format!("open failed: {e}"),
    })?;
    let mut reader = BufReader::new(file);

    // Collect (transform_type, params, fixed_params) tuples in file order.
    let mut entries: Vec<(String, Vec<f64>, Vec<f64>)> = Vec::new();
    loop {
        let Some(params_var) = read_variable(&mut reader, path)? else {
            break;
        };
        let fixed_var = read_variable(&mut reader, path)?.ok_or_else(|| XfmError::InvalidFile {
            path: path.to_path_buf(),
            reason: format!(
                "transform '{}' missing the trailing 'fixed' parameter variable",
                params_var.name
            ),
        })?;
        entries.push((params_var.name, params_var.values, fixed_var.values));
    }

    if entries.is_empty() {
        return Err(XfmError::InvalidFile {
            path: path.to_path_buf(),
            reason: "no transform components found in .mat file".into(),
        });
    }

    // ITK CompositeTransform applies its queue last-first (see itk_h5.rs and
    // itk_txt.rs for the same rationale); push entries in *reverse* file
    // order to align with our chain's stored-order apply semantics.
    let mut chain = TransformChain::new();
    for (ttype, params, fixed) in entries.into_iter().rev() {
        if !is_supported_affine_type(&ttype) {
            return Err(XfmError::UnsupportedTransformType(ttype));
        }
        chain.push_affine(build_affine(&ttype, &params, &fixed)?);
    }
    Ok(chain)
}

fn is_supported_affine_type(ttype: &str) -> bool {
    ttype.starts_with("AffineTransform")
        || ttype.starts_with("MatrixOffsetTransformBase")
        || ttype.starts_with("ScaleSkewVersor3DTransform")
        || ttype.starts_with("Rigid3DTransform")
        || ttype.starts_with("Similarity3DTransform")
        || ttype.starts_with("Euler3DTransform")
        || ttype.starts_with("VersorRigid3DTransform")
        || ttype.starts_with("ScaleVersor3DTransform")
        || ttype.starts_with("FixedCenterOfRotationAffineTransform")
}

fn build_affine(ttype: &str, params: &[f64], fixed: &[f64]) -> Result<Affine3> {
    if params.len() != AFFINE_PARAM_COUNT {
        return Err(XfmError::MalformedParameters(format!(
            "{ttype} expects {AFFINE_PARAM_COUNT} parameters (9 matrix + 3 translation), got {}",
            params.len()
        )));
    }
    let center = match fixed.len() {
        0 => Vector3::zeros(),
        AFFINE_FIXED_PARAM_COUNT => Vector3::new(fixed[0], fixed[1], fixed[2]),
        n => {
            return Err(XfmError::MalformedParameters(format!(
                "{ttype} fixed parameters must have 0 or {AFFINE_FIXED_PARAM_COUNT} values, got {n}"
            )))
        }
    };
    let m = Matrix3::new(
        params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7],
        params[8],
    );
    let t = Vector3::new(params[9], params[10], params[11]);
    Ok(Affine3::from_itk_components(m, t, center))
}

/// Read one MATLAB v4 variable (header + name + payload). Returns `Ok(None)`
/// on a clean EOF before the next header — that's how we detect end-of-file.
fn read_variable<R: Read>(reader: &mut R, path: &Path) -> Result<Option<Variable>> {
    let mut header_bytes = [0u8; HEADER_BYTES];
    let n_read = read_full_or_eof(reader, &mut header_bytes)?;
    if n_read == 0 {
        return Ok(None);
    }
    if n_read != HEADER_BYTES {
        return Err(XfmError::InvalidFile {
            path: path.to_path_buf(),
            reason: format!("truncated header (got {n_read} of {HEADER_BYTES} bytes)"),
        });
    }

    let header = decode_header(&header_bytes, path)?;
    if header.cols != 1 {
        return Err(XfmError::MalformedParameters(format!(
            "MATLAB variable has cols={}; ITK only writes column vectors",
            header.cols
        )));
    }

    // Name: namlen bytes including the trailing NUL (per VNL writer).
    let mut name_bytes = vec![0u8; header.namlen];
    reader.read_exact(&mut name_bytes).map_err(|e| XfmError::InvalidFile {
        path: path.to_path_buf(),
        reason: format!("truncated variable name (namlen={}): {e}", header.namlen),
    })?;
    let name = std::str::from_utf8(name_bytes.split(|&b| b == 0).next().unwrap_or(&[]))
        .map_err(|_| XfmError::InvalidFile {
            path: path.to_path_buf(),
            reason: "MATLAB variable name is not valid UTF-8".into(),
        })?
        .to_string();

    // Numeric payload.
    let bytes_per_value = if header.is_single_precision { 4 } else { 8 };
    let mut payload = vec![0u8; header.rows * bytes_per_value];
    reader.read_exact(&mut payload).map_err(|e| XfmError::InvalidFile {
        path: path.to_path_buf(),
        reason: format!("truncated payload for variable '{name}': {e}"),
    })?;

    let values = if header.is_single_precision {
        decode_floats(&payload, header.needs_swap)
    } else {
        decode_doubles(&payload, header.needs_swap)
    };

    Ok(Some(Variable { name, values }))
}

/// Try to read `buf.len()` bytes; return how many were actually read. Zero
/// means a clean EOF on the very first byte, which is how we detect the end
/// of the variable stream.
fn read_full_or_eof<R: Read>(reader: &mut R, buf: &mut [u8]) -> Result<usize> {
    let mut filled = 0;
    while filled < buf.len() {
        match reader.read(&mut buf[filled..]) {
            Ok(0) => break,
            Ok(n) => filled += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(XfmError::Io(e)),
        }
    }
    Ok(filled)
}

fn decode_header(bytes: &[u8; HEADER_BYTES], path: &Path) -> Result<RawHeader> {
    // Read the `type` field as little-endian and check whether it's one of
    // VNL's recognised "native byte order" codes; if not, the rest of the
    // header is big-endian and needs swapping (per `vnl_matlab_read.cxx`).
    let raw_type = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    let needs_swap = !matches!(raw_type, 0 | 10 | 100 | 110 | 1000 | 1100 | 1110);

    let read_u32 = |offset: usize| -> u32 {
        let chunk: [u8; 4] = bytes[offset..offset + 4].try_into().unwrap();
        if needs_swap {
            u32::from_be_bytes(chunk)
        } else {
            u32::from_le_bytes(chunk)
        }
    };

    let type_field = read_u32(0);
    let rows = read_u32(4) as usize;
    let cols = read_u32(8) as usize;
    let imag = read_u32(12);
    let namlen = read_u32(16) as usize;

    if imag != 0 {
        return Err(XfmError::InvalidFile {
            path: path.to_path_buf(),
            reason: "MATLAB variable is complex; ITK transforms are real-only".into(),
        });
    }
    if namlen == 0 {
        return Err(XfmError::InvalidFile {
            path: path.to_path_buf(),
            reason: "MATLAB variable has zero-length name".into(),
        });
    }

    // Tens place of `type` encodes precision: 0 = double, 1 = single.
    let precision_digit = (type_field % 100) / 10;
    let is_single_precision = match precision_digit {
        0 => false,
        1 => true,
        _ => {
            return Err(XfmError::InvalidFile {
                path: path.to_path_buf(),
                reason: format!("unsupported MATLAB type field: {type_field}"),
            })
        }
    };

    Ok(RawHeader {
        rows,
        cols,
        namlen,
        is_single_precision,
        needs_swap,
    })
}

fn decode_doubles(bytes: &[u8], swap: bool) -> Vec<f64> {
    bytes
        .chunks_exact(8)
        .map(|c| {
            let arr: [u8; 8] = c.try_into().unwrap();
            if swap {
                f64::from_be_bytes(arr)
            } else {
                f64::from_le_bytes(arr)
            }
        })
        .collect()
}

fn decode_floats(bytes: &[u8], swap: bool) -> Vec<f64> {
    bytes
        .chunks_exact(4)
        .map(|c| {
            let arr: [u8; 4] = c.try_into().unwrap();
            let v = if swap {
                f32::from_be_bytes(arr)
            } else {
                f32::from_le_bytes(arr)
            };
            v as f64
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// Hand-construct a MATLAB v4 variable matching the layout produced by
    /// `vnl_matlab_write` for a 1-D vector of f64s.
    fn write_var(out: &mut Vec<u8>, name: &str, values: &[f64]) {
        // type=0 (LE, column-wise, double, machine 0)
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&(values.len() as u32).to_le_bytes()); // rows
        out.extend_from_slice(&1u32.to_le_bytes()); // cols
        out.extend_from_slice(&0u32.to_le_bytes()); // imag
        let namlen = (name.len() + 1) as u32;
        out.extend_from_slice(&namlen.to_le_bytes());
        out.extend_from_slice(name.as_bytes());
        out.push(0); // trailing NUL
        for v in values {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }

    fn make_mat(entries: &[(&str, &[f64], &[f64])]) -> NamedTempFile {
        let mut buf = Vec::new();
        for (ttype, params, fixed) in entries {
            write_var(&mut buf, ttype, params);
            write_var(&mut buf, "fixed", fixed);
        }
        let mut f = tempfile::Builder::new()
            .suffix(".mat")
            .tempfile()
            .unwrap();
        f.write_all(&buf).unwrap();
        f.flush().unwrap();
        f
    }

    #[test]
    fn parse_identity_affine() {
        let identity_params = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let zero_center = [0.0, 0.0, 0.0];
        let f = make_mat(&[(
            "AffineTransform_double_3_3",
            &identity_params,
            &zero_center,
        )]);

        let chain = read_itk_mat(f.path()).unwrap();
        assert_eq!(chain.components.len(), 1);
        assert_eq!(chain.map_point([3.0, -2.0, 7.5]), [3.0, -2.0, 7.5]);
    }

    #[test]
    fn parse_translation_in_lps() {
        // ITK translation [1, 2, 3] in LPS → RAS [-1, -2, 3].
        let params = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 3.0];
        let zero_center = [0.0, 0.0, 0.0];
        let f = make_mat(&[("AffineTransform_double_3_3", &params, &zero_center)]);

        let chain = read_itk_mat(f.path()).unwrap();
        let p = chain.map_point([0.0, 0.0, 0.0]);
        assert!((p[0] + 1.0).abs() < 1e-12);
        assert!((p[1] + 2.0).abs() < 1e-12);
        assert!((p[2] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn rejects_unsupported_transform_type() {
        let f = make_mat(&[(
            "DisplacementFieldTransform_double_3_3",
            &[0.0; 12],
            &[0.0; 3],
        )]);
        let err = read_itk_mat(f.path()).unwrap_err();
        assert!(matches!(err, XfmError::UnsupportedTransformType(_)));
    }

    #[test]
    fn rejects_truncated_after_params() {
        let mut buf = Vec::new();
        write_var(&mut buf, "AffineTransform_double_3_3", &[0.0; 12]);
        // No trailing 'fixed' variable.
        let mut f = tempfile::Builder::new()
            .suffix(".mat")
            .tempfile()
            .unwrap();
        f.write_all(&buf).unwrap();
        f.flush().unwrap();
        let err = read_itk_mat(f.path()).unwrap_err();
        assert!(matches!(err, XfmError::InvalidFile { .. }));
    }

    #[test]
    fn two_sections_compose_in_itk_apply_order() {
        // Two LPS translations [1,0,0] and [0,2,0] — RAS components flip to
        // (-1,0,0) and (0,-2,0). Translations commute so order doesn't bite,
        // but we check both components are present and the composition lands
        // on the expected RAS point.
        let p1 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0];
        let p2 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0];
        let zero = [0.0, 0.0, 0.0];
        let f = make_mat(&[
            ("AffineTransform_double_3_3", &p1, &zero),
            ("AffineTransform_double_3_3", &p2, &zero),
        ]);
        let chain = read_itk_mat(f.path()).unwrap();
        assert_eq!(chain.components.len(), 2);
        let p = chain.map_point([0.0, 0.0, 0.0]);
        assert!((p[0] + 1.0).abs() < 1e-12);
        assert!((p[1] + 2.0).abs() < 1e-12);
    }
}
