#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch.nn.functional import *

from .activation import softmax_multidim
from .crop import crop_dim, crop_dims
from .get import get_device
from .indices import (
    get_inverse_perm,
    get_perm_indices,
    insert_at_indices,
    randperm_diff,
    remove_at_indices,
)
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
    onehot_to_index,
    onehot_to_name,
    probs_to_index,
    probs_to_name,
    probs_to_onehot,
)
from .multilabel import (
    indices_to_multihot,
    indices_to_names,
    multihot_to_indices,
    multihot_to_names,
    names_to_indices,
    names_to_multihot,
    probs_to_indices,
    probs_to_multihot,
    probs_to_names,
)
from .numpy import numpy_to_tensor, tensor_to_numpy, to_numpy
from .others import (
    can_be_converted_to_tensor,
    can_be_stacked,
    count_parameters,
    find,
    item,
    move_to_rec,
    ndim,
    shape,
)
from .pad import cat_padded_batch, pad_and_stack_rec, pad_dim, pad_dims
from .transform import (
    repeat_interleave_nd,
    resample_nearest_freqs,
    resample_nearest_rates,
    resample_nearest_steps,
    transform_drop,
)
