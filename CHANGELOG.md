# Change log

All notable changes to this project will be documented in this file.

## [0.4.1] UNRELEASED
### Changed
- Rename `is_pickle_root` to `is_packed_root` (old name was kept fpr backward compatibility).


## [0.4.0] 2024-05-27
### Added
- Option `subdir_size` to `pack_dataset`.
- Classes `CosDecayScheduler` and `CosDecayRule` classes.
- Function `sorted_dict` to collections.
- Functions `ndim` and `shape` to search information in tensor-like objects.
- Function `item` to convert scalar-like objects to built-in scalar objects.
- Mixin classes to improve module features.

### Modified
- Rename `from_numpy` to `numpy_to_tensor` to avoid confusion with `torch.from_numpy`.
- Rename `save_to_yaml` to `to_yaml` and `save_to_csv` to `to_csv`.
- Rename `PickleDataset` to `PackedDataset`.
- Rename `pack_to_pickle` to `pack_dataset`.
- Update `EModule` and `ESequential` classes with auto-config and auto-device detection.

### Fixed
- Function `can_be_converted_to_tensor` now accepts numpy arrays and scalars.
- Wildcard imports, and module imports. (e.g. `from torchoutil import nn`)
- `can_be_stacked` now returns False with an empty sequence.


## [0.3.1] 2024-04-25
### Added
- Method `count_parameters` to `TModule`.
- Option `padding_idx` to `indices_to_onehot` function.
- Functions `dict_list_to_list_dict`, `flat_list`, `unflat_dict_of_dict` to collections utils.
- Class `PickleDataset` and function `pack_to_pickle` to utils.
- Class `ResampleNearest` and function `resample_nearest`.
- Class `TransformDrop` and function `transform_drop`.

### Fixed
- Remove invalid test file.
- Function `is_scalar` now returns True for numpy scalars when numpy package is installed.


## [0.3.0] 2024-04-17
### Added
- `PositionalEncoding` layer for transformers networks.
- Property `metadata` to `HDFDataset`.
- Function `create_params_groups_bias` for avoid apply weight decay to networks bias.
- Functions `is_numpy_scalar`, `is_torch_scalar` and `unzip`.
- Options `auto_open`, `numpy_to_torch` and `file_kwargs` to customize loading in `HDFDataset`.

### Modified
- Update keyword arguments for mask, pad, labels functions and modules.
- Getting added column using `at` from `HDFDataset` when `return_added_columns` is `False`.
- Renamed `ModelCheckpointRegister` to `RegistryHub` and add hash value check after download.

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
