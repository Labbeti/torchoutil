#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch.nn.functional import *

from .activation import log_softmax_multidim, softmax_multidim
from .checksum import checksum, checksum_any
from .crop import crop_dim, crop_dims
from .indices import (
    get_inverse_perm,
    get_perm_indices,
    insert_at_indices,
    randperm_diff,
    remove_at_indices,
)
from .make import make_device, make_dtype, make_generator
from .mask import (
    generate_square_subsequent_mask,
    lengths_to_non_pad_mask,
    lengths_to_pad_mask,
    masked_equal,
    masked_mean,
    masked_sum,
    non_pad_mask_to_lengths,
    pad_mask_to_lengths,
    tensor_to_lengths,
    tensor_to_non_pad_mask,
    tensor_to_pad_mask,
    tensor_to_tensors_list,
    tensors_list_to_lengths,
)
from .multiclass import (
    index_to_name,
    index_to_onehot,
    name_to_index,
    name_to_onehot,
    one_hot,
    onehot_to_index,
    onehot_to_name,
    probs_to_index,
    probs_to_name,
    probs_to_onehot,
)
from .multilabel import (
    indices_to_multihot,
    indices_to_multinames,
    multihot_to_indices,
    multihot_to_multinames,
    multinames_to_indices,
    multinames_to_multihot,
    probs_to_indices,
    probs_to_multihot,
    probs_to_multinames,
)
from .numpy import numpy_to_tensor, tensor_to_numpy, to_numpy
from .others import (
    average_power,
    count_parameters,
    find,
    mse,
    ndim,
    nelement,
    prod,
    ranks,
    rmse,
    shape,
)
from .pad import cat_padded_batch, pad_and_stack_rec, pad_dim, pad_dims
from .powerset import multilabel_to_powerset, powerset_to_multilabel
from .predicate import (
    all_eq,
    all_ne,
    can_be_converted_to_tensor,
    can_be_stacked,
    is_complex,
    is_floating_point,
    is_full,
    is_sorted,
    is_unique,
)
from .segments import (
    activity_to_segments,
    activity_to_segments_list,
    extract_segments,
    segments_list_to_activity,
    segments_to_activity,
    segments_to_list,
    segments_to_segments_list,
)
from .transform import (
    flatten,
    identity,
    move_to_rec,
    pad_and_crop_dim,
    repeat_interleave_nd,
    resample_nearest_freqs,
    resample_nearest_rates,
    resample_nearest_steps,
    shuffled,
    squeeze,
    squeeze_,
    squeeze_copy,
    to_item,
    to_tensor,
    transform_drop,
    unsqueeze,
    unsqueeze_,
    unsqueeze_copy,
    view_as_complex,
    view_as_real,
)
