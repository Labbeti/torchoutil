# Change log

All notable changes to this project will be documented in this file.

## [0.2.1] UNRELEASED
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
