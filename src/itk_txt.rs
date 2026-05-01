//! Reader for the **Insight Transform File V1.0** plain-text format used
//! by ITK and ANTs for affine transforms. Format is the same parameter
//! convention as `*Composite.h5`'s `AffineTransform_*_3_3` entries — 12
//! parameters (9 row-major matrix + 3 translation) plus 3 fixed
//! parameters (center of rotation) — but encoded as ASCII rather than
//! HDF5.
//!
//! The format is **affine-only by spec**; displacement-field components
//! are returned by [`read_itk_h5`](crate::read_itk_h5) only and surface
//! as [`XfmError::UnsupportedTransformType`] here.
//!
//! Example file:
//!
//! ```text
//! #Insight Transform File V1.0
//! #Transform 0
//! Transform: MatrixOffsetTransformBase_double_3_3
//! Parameters: 1.06 -0.0066 -0.0071 0.034 1.03 0.00005 0.0056 0.046 1.04 -0.23 -4.35 2.80
//! FixedParameters: 2.14 4.70 24.39
//! ```
//!
//! The file may declare multiple `#Transform N` sections; they're
//! returned in order as a multi-component [`TransformChain`], applied as
//! `chain(p) = c_N(c_{N-1}(... c_1(p)))` matching the same composition
//! semantics as the h5 reader.

use std::fs;
use std::path::Path;

use nalgebra::{Matrix3, Vector3};

use crate::affine::Affine3;
use crate::chain::TransformChain;
use crate::error::{Result, XfmError};

/// Read an Insight Transform File V1.0 (`.txt`) and return a
/// [`TransformChain`] in RAS+ mm.
pub fn read_itk_txt(path: &Path) -> Result<TransformChain> {
    let body = fs::read_to_string(path).map_err(|e| XfmError::InvalidFile {
        path: path.to_path_buf(),
        reason: format!("read failed: {e}"),
    })?;
    parse(&body, path)
}

#[derive(Default)]
struct Section {
    transform_type: Option<String>,
    parameters: Option<Vec<f64>>,
    fixed_parameters: Option<Vec<f64>>,
}

impl Section {
    fn is_empty(&self) -> bool {
        self.transform_type.is_none()
            && self.parameters.is_none()
            && self.fixed_parameters.is_none()
    }
}

fn parse(body: &str, path: &Path) -> Result<TransformChain> {
    // Verify the magic header.
    let mut lines = body.lines();
    match lines.next() {
        Some(first) if first.trim_start().starts_with("#Insight Transform File") => {}
        _ => {
            return Err(XfmError::InvalidFile {
                path: path.to_path_buf(),
                reason: "missing '#Insight Transform File' magic".into(),
            })
        }
    }

    // Walk remaining lines, splitting on `#Transform N` boundaries.
    let mut sections: Vec<Section> = Vec::new();
    let mut current = Section::default();
    for line in lines {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.starts_with("#Transform") {
            if !current.is_empty() {
                sections.push(std::mem::take(&mut current));
            }
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("Transform:") {
            current.transform_type = Some(rest.trim().to_string());
        } else if let Some(rest) = trimmed.strip_prefix("Parameters:") {
            current.parameters = Some(parse_floats(rest));
        } else if let Some(rest) = trimmed.strip_prefix("FixedParameters:") {
            current.fixed_parameters = Some(parse_floats(rest));
        } else if trimmed.starts_with('#') {
            // Other comments are ignored.
        }
    }
    if !current.is_empty() {
        sections.push(current);
    }

    if sections.is_empty() {
        return Err(XfmError::InvalidFile {
            path: path.to_path_buf(),
            reason: "no #Transform sections in file".into(),
        });
    }

    let mut chain = TransformChain::new();
    for (idx, section) in sections.into_iter().enumerate() {
        let ttype = section
            .transform_type
            .ok_or_else(|| XfmError::InvalidFile {
                path: path.to_path_buf(),
                reason: format!("section {idx} is missing 'Transform:' line"),
            })?;
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
            let params = section.parameters.ok_or_else(|| {
                XfmError::MalformedParameters(format!(
                    "section {idx} '{ttype}' missing 'Parameters:' line"
                ))
            })?;
            let fixed = section.fixed_parameters.unwrap_or_default();
            chain.push_affine(build_affine(&ttype, &params, &fixed)?);
        } else {
            return Err(XfmError::UnsupportedTransformType(ttype));
        }
    }
    Ok(chain)
}

fn build_affine(ttype: &str, params: &[f64], fixed: &[f64]) -> Result<Affine3> {
    if params.len() != 12 {
        return Err(XfmError::MalformedParameters(format!(
            "{ttype} expects 12 Parameters (9 matrix + 3 translation), got {}",
            params.len()
        )));
    }
    let center = match fixed.len() {
        0 => Vector3::zeros(),
        3 => Vector3::new(fixed[0], fixed[1], fixed[2]),
        n => {
            return Err(XfmError::MalformedParameters(format!(
                "{ttype} FixedParameters must have 0 or 3 floats, got {n}"
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

fn parse_floats(s: &str) -> Vec<f64> {
    s.split_whitespace()
        .filter_map(|tok| tok.parse::<f64>().ok())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn write_temp(body: &str) -> NamedTempFile {
        let mut f = tempfile::Builder::new().suffix(".txt").tempfile().unwrap();
        f.write_all(body.as_bytes()).unwrap();
        f.flush().unwrap();
        f
    }

    #[test]
    fn parse_zero_center_identity() {
        let body = "\
#Insight Transform File V1.0
#Transform 0
Transform: AffineTransform_double_3_3
Parameters: 1 0 0 0 1 0 0 0 1 0 0 0
FixedParameters: 0 0 0
";
        let f = write_temp(body);
        let chain = read_itk_txt(f.path()).unwrap();
        assert_eq!(chain.components.len(), 1);
        // Identity round-trips any point.
        let p = chain.map_point([3.0, -1.0, 7.5]);
        assert!((p[0] - 3.0).abs() < 1e-12);
        assert!((p[1] + 1.0).abs() < 1e-12);
        assert!((p[2] - 7.5).abs() < 1e-12);
    }

    #[test]
    fn parse_translation_in_lps() {
        // ITK translation [1, 2, 3] in LPS → RAS [-1, -2, 3].
        let body = "\
#Insight Transform File V1.0
#Transform 0
Transform: AffineTransform_double_3_3
Parameters: 1 0 0 0 1 0 0 0 1 1 2 3
FixedParameters: 0 0 0
";
        let f = write_temp(body);
        let chain = read_itk_txt(f.path()).unwrap();
        let p = chain.map_point([0.0, 0.0, 0.0]);
        assert!((p[0] + 1.0).abs() < 1e-12);
        assert!((p[1] + 2.0).abs() < 1e-12);
        assert!((p[2] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn parse_two_sections_in_order() {
        // Two translations that should compose: [1,0,0] then [0,2,0],
        // both in LPS → [-1,-2,0] in RAS+ when applied to origin.
        let body = "\
#Insight Transform File V1.0
#Transform 0
Transform: AffineTransform_double_3_3
Parameters: 1 0 0 0 1 0 0 0 1 1 0 0
FixedParameters: 0 0 0
#Transform 1
Transform: AffineTransform_double_3_3
Parameters: 1 0 0 0 1 0 0 0 1 0 2 0
FixedParameters: 0 0 0
";
        let f = write_temp(body);
        let chain = read_itk_txt(f.path()).unwrap();
        assert_eq!(chain.components.len(), 2);
        let p = chain.map_point([0.0, 0.0, 0.0]);
        assert!((p[0] + 1.0).abs() < 1e-12);
        assert!((p[1] + 2.0).abs() < 1e-12);
    }

    #[test]
    fn rejects_displacement_field_type() {
        let body = "\
#Insight Transform File V1.0
#Transform 0
Transform: DisplacementFieldTransform_double_3_3
Parameters: 1 0 0 0 1 0 0 0 1 0 0 0
FixedParameters: 0 0 0
";
        let f = write_temp(body);
        let err = read_itk_txt(f.path()).unwrap_err();
        assert!(matches!(err, XfmError::UnsupportedTransformType(_)));
    }

    #[test]
    fn rejects_missing_magic() {
        let body = "Parameters: 1 0 0 0 1 0 0 0 1 0 0 0\n";
        let f = write_temp(body);
        let err = read_itk_txt(f.path()).unwrap_err();
        assert!(matches!(err, XfmError::InvalidFile { .. }));
    }

    #[test]
    fn rejects_wrong_parameter_count() {
        let body = "\
#Insight Transform File V1.0
#Transform 0
Transform: AffineTransform_double_3_3
Parameters: 1 0 0 0 1 0 0 0 1 0 0
FixedParameters: 0 0 0
";
        let f = write_temp(body);
        let err = read_itk_txt(f.path()).unwrap_err();
        assert!(matches!(err, XfmError::MalformedParameters(_)));
    }
}
