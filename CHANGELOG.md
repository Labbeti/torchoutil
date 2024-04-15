# Change log

All notable changes to this project will be documented in this file.

## [0.3.0] UNRELEASED
### Added
- `PositionalEncoding` layer for transformers networks.
- Property `metadata` to `HDFDataset`.
- Function `create_params_groups_bias` for avoid apply weight decay to networks bias.
- Functions `is_numpy_scalar`, `is_torch_scalar` and `unzip`.

### Modified
- Update keyword arguments for mask, pad, labels functions and modules.
- Getting added column using `at` from `HDFDataset` when `return_added_columns` is `False`.

### Fixed
- `can_be_converted_to_tensor` now returns True if input is a Tensor.

## [0.2.2] 2024-03-08
### Fixed
- `ModelCheckpointRegister` now creates intermediate directories before download.
- `MaskedMean` and `MaskedSum` dim argument.


## [0.2.1] 2024-03-07
### Added
- `ModelCheckpointRegister` class to make download and loading easier from checkpoint.

### Modified
- `pack_to_hdf` now supports existing shape column.
- `SizedDatasetLike` is now compatible with `Sequence`-like objects.


## [0.2.0] 2024-03-04
### Added
- Optional hdf datasets with `HDFDataset` class and `pack_to_hdf` function.
- Multiclass functions to convert labels.
- Arg `diagonal` to `generate_square_subsequent_mask`.
- `Abs`, `Angle`, `Real` and `Imag` modules.

### Modified
- Use Literal for pad and crop functions typing.
- Minimal torch version is now 1.10.


## [0.1.0] 2024-01-29
### Added
- 47 torch functions and 40 torch modules.
