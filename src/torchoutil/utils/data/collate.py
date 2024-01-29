#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Dict, Iterable, List, Optional

import torch

from torchoutil.nn.functional.others import can_be_converted_to_tensor, can_be_stacked
from torchoutil.nn.functional.pad import pad_and_stack_rec
from torchoutil.utils.collections import filter_iterable, list_dict_to_dict_list


class CollateDict:
    """Collate list of dict into a dict of list WITHOUT auto-padding."""

    def __init__(self, error_on_missing_key: bool = True) -> None:
        super().__init__()
        self.error_on_missing_key = error_on_missing_key

    def __call__(self, batch_lst: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        batch_dict = list_dict_to_dict_list(
            batch_lst, error_on_missing_key=self.error_on_missing_key
        )
        return batch_dict


class AdvancedCollateDict:
    def __init__(
        self,
        pad_values: Optional[Dict[str, Any]] = None,
        include_keys: Optional[Iterable[str]] = None,
        exclude_keys: Optional[Iterable[str]] = None,
    ) -> None:
        """Collate list of dict into a dict of list WITH auto-padding for given keys."""
        if pad_values is None:
            pad_values = {}

        super().__init__()
        self.pad_values = pad_values
        self.include_keys = include_keys
        self.exclude_keys = exclude_keys

    def __call__(self, batch_lst: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch_dict = list_dict_to_dict_list(batch_lst, error_on_missing_key=True)
        batch_keys = filter_iterable(
            batch_dict.keys(), self.include_keys, self.exclude_keys
        )
        batch_dict = {k: batch_dict[k] for k in batch_keys}

        result = {}

        for key, values in batch_dict.items():
            if key in self.pad_values:
                values = pad_and_stack_rec(values, self.pad_values[key])
            elif can_be_stacked(values):
                values = torch.stack(values)
            elif can_be_converted_to_tensor(values):
                values = torch.as_tensor(values)

            result[key] = values

        return result
