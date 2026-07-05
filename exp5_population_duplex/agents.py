"""Independent population agents for Experiment 5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Categorical
except ImportError as exc:  # pragma: no cover - exercised only without torch
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    Categorical = None  # type: ignore
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

_BaseModule = nn.Module if nn is not None else object

from .messages import EOS, FIRST_CONTENT_TOKEN, PAD


def require_torch() -> None:
    if _TORCH_IMPORT_ERROR is not None:
        raise ImportError(
            "Experiment 5 training/evaluation requires PyTorch. Install CPU PyTorch with "
            "`pip install torch --index-url https://download.pytorch.org/whl/cpu`."
        ) from _TORCH_IMPORT_ERROR


@dataclass
class ModelConfig:
    num_colors: int = 4
    num_shapes: int = 4
    num_sizes: int = 3
    num_candidates: int = 6
    vocabulary_size: int = 64
    attribute_embedding_dim: int = 32
    token_embedding_dim: int = 32
    hidden_dim: int = 128
    gru_layers: int = 1


class DuplexPopulationAgent(_BaseModule):  # type: ignore[misc]
    """One independently parameterized agent capable of both roles."""

    def __init__(self, config: ModelConfig):
        require_torch()
        super().__init__()
        self.config = config

        e_attr = config.attribute_embedding_dim
        h = config.hidden_dim
        e_tok = config.token_embedding_dim

        self.color_emb = nn.Embedding(config.num_colors, e_attr)
        self.shape_emb = nn.Embedding(config.num_shapes, e_attr)
        self.size_emb = nn.Embedding(config.num_sizes, e_attr)
        self.object_encoder = nn.Sequential(
            nn.Linear(e_attr * 3, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
        )

        self.token_emb = nn.Embedding(config.vocabulary_size, e_tok, padding_idx=PAD)
        self.message_encoder = nn.GRU(
            input_size=e_tok,
            hidden_size=h,
            num_layers=config.gru_layers,
            batch_first=True,
        )

        self.initial_decoder = nn.Linear(h, config.vocabulary_size)
        self.feedback_decoder = nn.Linear(h * 2, config.vocabulary_size)
        self.response_decoder = nn.Linear(h * 3, config.vocabulary_size)
        self.selection_head = nn.Sequential(nn.Linear(h * 4, h), nn.ReLU(), nn.Linear(h, 1))

        self.initial_value = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
        self.feedback_value = nn.Sequential(nn.Linear(h * 2, h), nn.ReLU(), nn.Linear(h, 1))
        self.response_value = nn.Sequential(nn.Linear(h * 3, h), nn.ReLU(), nn.Linear(h, 1))
        self.selection_value = nn.Sequential(nn.Linear(h * 3, h), nn.ReLU(), nn.Linear(h, 1))

        self.position_token_bias = nn.Parameter(torch.randn(8, config.vocabulary_size) * 0.02)

    def encode_object(self, obj: "torch.Tensor") -> "torch.Tensor":
        c = self.color_emb(obj[..., 0])
        s = self.shape_emb(obj[..., 1])
        z = self.size_emb(obj[..., 2])
        return self.object_encoder(torch.cat([c, s, z], dim=-1))

    def encode_world(self, objects: "torch.Tensor") -> "torch.Tensor":
        encoded = self.encode_object(objects)
        return encoded.mean(dim=-2)

    def encode_message(self, message: "torch.Tensor") -> "torch.Tensor":
        if message.numel() == 0:
            return torch.zeros(self.config.hidden_dim, device=message.device)
        emb = self.token_emb(message.long().unsqueeze(0))
        _, h_n = self.message_encoder(emb)
        return h_n[-1, 0]

    def _repeat_logits(self, context: "torch.Tensor", decoder: "nn.Linear", max_len: int) -> "torch.Tensor":
        if max_len == 0:
            return torch.empty(0, self.config.vocabulary_size, device=context.device)
        states = context.unsqueeze(0).repeat(max_len, 1)
        return decoder(states) + self.position_token_bias[:max_len]

    def initial_logits(self, target: "torch.Tensor", max_len: int) -> tuple["torch.Tensor", "torch.Tensor"]:
        ctx = self.encode_object(target)
        return self._repeat_logits(ctx, self.initial_decoder, max_len), self.initial_value(ctx).squeeze(-1)

    def feedback_logits(
        self,
        world: "torch.Tensor",
        initial_message: "torch.Tensor",
        max_len: int,
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        ctx = torch.cat([self.encode_world(world), self.encode_message(initial_message)], dim=-1)
        return self._repeat_logits(ctx, self.feedback_decoder, max_len), self.feedback_value(ctx).squeeze(-1)

    def response_logits(
        self,
        target: "torch.Tensor",
        initial_message: "torch.Tensor",
        feedback_message: "torch.Tensor",
        max_len: int,
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        ctx = torch.cat(
            [
                self.encode_object(target),
                self.encode_message(initial_message),
                self.encode_message(feedback_message),
            ],
            dim=-1,
        )
        return self._repeat_logits(ctx, self.response_decoder, max_len), self.response_value(ctx).squeeze(-1)

    def selection_logits(
        self,
        world: "torch.Tensor",
        initial_message: "torch.Tensor",
        feedback_message: "torch.Tensor",
        response_message: "torch.Tensor",
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        msg_ctx = torch.cat(
            [
                self.encode_message(initial_message),
                self.encode_message(feedback_message),
                self.encode_message(response_message),
            ],
            dim=-1,
        )
        object_repr = self.encode_object(world)
        msg_expand = msg_ctx.unsqueeze(0).expand(object_repr.shape[0], -1)
        logits = self.selection_head(torch.cat([object_repr, msg_expand], dim=-1)).squeeze(-1)
        value = self.selection_value(msg_ctx).squeeze(-1)
        return logits, value


def _active_logits(logits: "torch.Tensor", vocabulary_size: int) -> "torch.Tensor":
    masked = logits[:vocabulary_size].clone()
    masked[PAD] = -1e9
    return masked


def sample_message(
    logits: "torch.Tensor",
    vocabulary_size: int,
    greedy: bool = False,
) -> Dict[str, "torch.Tensor"]:
    """Sample or greedily decode a variable-length message.

    PAD is never sampled while active. Once EOS is produced, the remaining
    positions become PAD and do not add log-probability or entropy.
    """

    require_torch()
    tokens: List["torch.Tensor"] = []
    log_probs: List["torch.Tensor"] = []
    entropies: List["torch.Tensor"] = []
    active = True
    device = logits.device

    for pos in range(logits.shape[0]):
        if not active:
            tokens.append(torch.tensor(PAD, dtype=torch.long, device=device))
            continue

        dist = Categorical(logits=_active_logits(logits[pos], vocabulary_size))
        token = torch.argmax(dist.logits) if greedy else dist.sample()
        tokens.append(token.long())
        log_probs.append(dist.log_prob(token))
        entropies.append(dist.entropy())
        if int(token.item()) == EOS:
            active = False

    if logits.shape[0] == 0:
        message = torch.empty(0, dtype=torch.long, device=device)
    else:
        message = torch.stack(tokens)
    zero = torch.tensor(0.0, device=device)
    return {
        "message": message,
        "log_prob": torch.stack(log_probs).sum() if log_probs else zero,
        "entropy": torch.stack(entropies).sum() if entropies else zero,
    }


def message_content_length(message: "torch.Tensor") -> int:
    return int((message >= FIRST_CONTENT_TOKEN).sum().item())


def build_population(config: ModelConfig, population_size: int, device: "torch.device") -> List[DuplexPopulationAgent]:
    require_torch()
    return [DuplexPopulationAgent(config).to(device) for _ in range(population_size)]


def empty_message(max_len: int, device: "torch.device") -> "torch.Tensor":
    if max_len == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    tokens = [EOS] + [PAD] * (max_len - 1)
    return torch.tensor(tokens, dtype=torch.long, device=device)
