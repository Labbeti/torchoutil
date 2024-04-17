#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Dict, Iterable, List, Optional, TypeVar

import torch

from torchoutil.nn.functional.others import can_be_converted_to_tensor, can_be_stacked
from torchoutil.nn.functional.pad import pad_and_stack_rec
from torchoutil.utils.collections import (
    KeyMode,
    filter_iterable,
    list_dict_to_dict_list,
)

K = TypeVar("K")
V = TypeVar("V")


class CollateDict:
    """Collate class to handle a batch dict."""

    def __init__(self, key_mode: KeyMode = "same") -> None:
        super().__init__()
        self.key_mode: KeyMode = key_mode

    def __call__(self, batch_lst: List[Dict[K, V]]) -> Dict[K, List[V]]:
        result = list_dict_to_dict_list(
            batch_lst,
            key_mode=self.key_mode,
        )
        return result  # type: ignore


class AdvancedCollateDict:
    """Collate class to automatically convert to tensor, pad sequences and filter keys in a batch dict."""

    def __init__(
        self,
        pad_values: Optional[Dict[str, Any]] = None,
        include_keys: Optional[Iterable[str]] = None,
        exclude_keys: Optional[Iterable[str]] = None,
        key_mode: KeyMode = "same",
    ) -> None:
        """Collate list of dict into a dict of list WITH auto-padding for given keys."""
        if pad_values is None:
            pad_values = {}

        super().__init__()
        self.pad_values = pad_values
        self.include_keys = include_keys
        self.exclude_keys = exclude_keys
        self.key_mode: KeyMode = key_mode

    def __call__(self, batch_lst: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch_dict = list_dict_to_dict_list(
            batch_lst,
            key_mode=self.key_mode,
        )
        batch_keys = filter_iterable(
            batch_dict.keys(),
            self.include_keys,
            self.exclude_keys,
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
