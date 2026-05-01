# nitransforms fixture

`affine-antsComposite.h5` is vendored verbatim from the
[nitransforms](https://github.com/nipy/nitransforms) test data:

- Path upstream: `nitransforms/tests/data/affine-antsComposite.h5`

It is a canonical ANTs `antsRegistration` Composite.h5 output containing a
single `AffineTransform_double_3_3` component, used by upstream
nitransforms' own ITK round-trip tests. Here it exercises
[`read_itk_h5`](../../../src/itk_h5.rs) end-to-end against the same
matrix and point values nitransforms computes in Python.

Upstream nitransforms is licensed under the MIT License (Copyright (c)
2021 The NiPy developers). The file is redistributed unmodified.
