#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math

import torch
from torch import Tensor, nn
from torch.types import Device

from torchoutil.nn.functional.get import get_device


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        emb_size: int,
        dropout_p: float,
        maxlen: int = 5000,
        device: Device = None,
    ) -> None:
        """Vanilla Positional Encoding for transformers networks.

        Based on PyTorch tutorial from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        """
        pos_embedding = init_pos_emb(emb_size, maxlen, device)

        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.register_buffer("pos_embedding", pos_embedding)
        self.pos_embedding: Tensor

    def forward(self, token_emb: Tensor) -> Tensor:
        pos_emb = self.pos_embedding[: token_emb.size(0), :]
        output = self.dropout(token_emb + pos_emb)
        return output


def init_pos_emb(
    emb_size: int,
    maxlen: int = 5000,
    device: Device = None,
) -> Tensor:
    """Returns positional embedding tensor of shape (1, maxlen, emb_size)."""
    device = get_device(device)

    den = torch.exp(
        -torch.arange(0, emb_size, 2, device=device) * math.log(10000) / emb_size
    )
    pos = torch.arange(0, maxlen, device=device).reshape(maxlen, 1)

    pos_embedding = torch.zeros((maxlen, emb_size))
    pos_embedding[:, 0::2] = torch.sin(pos * den)
    pos_embedding[:, 1::2] = torch.cos(pos * den)
    pos_embedding = pos_embedding.unsqueeze(-2)

    return pos_embedding
