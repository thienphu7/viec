"""Clean model definitions corresponding to the ViEC paper architecture family."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


@dataclass
class ModelOutput:
    """Forward output used by training and evaluation code."""

    intent_logits: torch.Tensor
    slot_logits: torch.Tensor | None = None
    mask: torch.Tensor | None = None


class BaselineCNN(nn.Module):
    """Sentence-level convolutional baseline for intent detection."""

    def __init__(
        self,
        input_dim: int,
        num_intents: int,
        channels: int,
        kernel_sizes: tuple[int, ...] = (3, 4, 5),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            nn.Conv1d(input_dim, channels, kernel_size=k, padding=k // 2) for k in kernel_sizes
        )
        self.dropout = nn.Dropout(dropout)
        self.intent_head = nn.Linear(channels * len(kernel_sizes), num_intents)

    def forward(self, features: torch.Tensor, mask: torch.Tensor | None = None) -> ModelOutput:
        x = features.transpose(1, 2)
        pooled = []
        for conv in self.convs:
            y = torch.relu(conv(x))
            if mask is not None:
                y = y.masked_fill(~mask.unsqueeze(1), float("-inf"))
            pooled.append(torch.max(y, dim=-1).values)
        sentence = self.dropout(torch.cat(pooled, dim=-1))
        return ModelOutput(intent_logits=self.intent_head(sentence), mask=mask)


class HierarchicalCNN(nn.Module):
    """Token-level convolutional encoder used inside the hierarchical backbone."""

    def __init__(self, input_dim: int, channels: int, kernel_sizes: tuple[int, ...] = (3, 5)) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            nn.Conv1d(input_dim, channels, kernel_size=k, padding=k // 2) for k in kernel_sizes
        )
        self.output_dim = channels * len(kernel_sizes)

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        del lengths
        x = features.transpose(1, 2)
        outputs = [torch.relu(conv(x)).transpose(1, 2) for conv in self.convs]
        return torch.cat(outputs, dim=-1)


class HierarchicalBiLSTM(nn.Module):
    """Bidirectional LSTM token encoder."""

    def __init__(self, input_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_size, batch_first=True, bidirectional=True)
        self.output_dim = hidden_size * 2

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(
            features, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        encoded, _ = self.rnn(packed)
        padded, _ = pad_packed_sequence(encoded, batch_first=True, total_length=features.size(1))
        return padded


class HierarchicalBiGRU(nn.Module):
    """Bidirectional GRU token encoder."""

    def __init__(self, input_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_size, batch_first=True, bidirectional=True)
        self.output_dim = hidden_size * 2

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(
            features, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        encoded, _ = self.rnn(packed)
        padded, _ = pad_packed_sequence(encoded, batch_first=True, total_length=features.size(1))
        return padded


class HierarchicalIntentSlotModel(nn.Module):
    """Shared token encoder with intent and optional slot heads.

    This class captures the multi-task hierarchical flow used in the notebooks:
    token features -> shared encoder -> masked sentence pooling -> intent head
    and token-level slot emissions. CRF decoding/loss is handled outside the
    module to keep the architecture inspectable.
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_intents: int,
        num_slots: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.intent_head = nn.Linear(encoder.output_dim, num_intents)
        self.slot_head = nn.Linear(encoder.output_dim, num_slots) if num_slots else None

    def forward(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> ModelOutput:
        encoded = self.dropout(self.encoder(features, lengths))
        if mask is None:
            steps = torch.arange(features.size(1), device=features.device).unsqueeze(0)
            mask = steps < lengths.unsqueeze(1)

        masked = encoded.masked_fill(~mask.unsqueeze(-1), 0.0)
        sentence = masked.sum(dim=1) / lengths.clamp(min=1).unsqueeze(-1)
        slot_logits = self.slot_head(encoded) if self.slot_head is not None else None
        return ModelOutput(
            intent_logits=self.intent_head(sentence),
            slot_logits=slot_logits,
            mask=mask,
        )


class CNNJointModel(HierarchicalIntentSlotModel):
    """Convenience wrapper for the hierarchical CNN variant."""

    def __init__(
        self,
        input_dim: int,
        num_intents: int,
        channels: int,
        num_slots: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        encoder = HierarchicalCNN(input_dim=input_dim, channels=channels)
        super().__init__(encoder, num_intents=num_intents, num_slots=num_slots, dropout=dropout)
