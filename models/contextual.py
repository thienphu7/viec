"""Contextual encoder branch for PhoBERT/XLM-R style models."""

from __future__ import annotations

import torch
from torch import nn


class ContextualFeatureBranch(nn.Module):
    """Project transformer subword states back to token-level features.

    The branch gathers the first subword hidden state for each whitespace token,
    masks truncated tokens, and projects transformer dimensionality to the
    downstream hierarchical model dimension. It is intentionally generic:
    callers pass an already constructed Hugging Face `AutoModel`.
    """

    def __init__(self, transformer: nn.Module, output_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_size = transformer.config.hidden_size
        self.transformer = transformer
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(hidden_size, output_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        word_positions: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state

        safe_positions = word_positions.clamp(min=0)
        gather_index = safe_positions.unsqueeze(-1).expand(-1, -1, hidden.size(-1))
        token_features = hidden.gather(dim=1, index=gather_index)
        token_mask = (word_positions >= 0).unsqueeze(-1).to(token_features.dtype)
        token_features = token_features * token_mask
        return self.projection(self.dropout(token_features))
