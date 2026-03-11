"""
Neural Speaker and Listener agents for the referential game.

Both agents are implemented in PyTorch and are compatible with simple
policy-gradient training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .communication import Vocabulary


@dataclass
class SpeakerConfig:
    """Configuration for the Speaker agent."""

    hidden_dim: int = 64


@dataclass
class ListenerConfig:
    """Configuration for the Listener agent."""

    hidden_dim: int = 64


class SpeakerAgent(nn.Module):
    """Speaker agent that maps a target object to a message.

    The Speaker observes a 3-dimensional integer vector (color, shape, size).
    It embeds each attribute and runs them through an MLP to produce logits
    over the vocabulary for each message position. Positions are conditionally
    independent given the target encoding, which keeps the architecture simple
    yet expressive enough for many emergent-communication experiments.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        message_length: int,
        world_color_dim: int,
        world_shape_dim: int,
        world_size_dim: int,
        config: SpeakerConfig | None = None,
    ) -> None:
        super().__init__()
        self.vocab = vocab
        self.message_length = message_length
        self.config = config or SpeakerConfig()

        embed_dim = self.config.hidden_dim

        self.color_emb = nn.Embedding(world_color_dim, embed_dim)
        self.shape_emb = nn.Embedding(world_shape_dim, embed_dim)
        self.size_emb = nn.Embedding(world_size_dim, embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 3, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
        )
        self.msg_head = nn.Linear(self.config.hidden_dim, self.message_length * self.vocab.size)

    def forward(self, obj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Produce a distribution over messages given an object.

        Args:
            obj: Tensor of shape (..., 3) with integer attributes.

        Returns:
            logits: Tensor of shape (..., message_length, vocab_size).
            probs: Softmax probabilities with the same shape.
        """
        color_idx = obj[..., 0]
        shape_idx = obj[..., 1]
        size_idx = obj[..., 2]

        c = self.color_emb(color_idx)
        s = self.shape_emb(shape_idx)
        z = self.size_emb(size_idx)
        x = torch.cat([c, s, z], dim=-1)

        h = self.mlp(x)
        logits = self.msg_head(h)
        logits = logits.view(*logits.shape[:-1], self.message_length, self.vocab.size)
        probs = F.softmax(logits, dim=-1)
        return logits, probs

    def act(self, obj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a message and return its log-probability.

        Args:
            obj: Tensor of shape (3,) with integer attributes.

        Returns:
            message: Tensor of shape (message_length,) with token indices.
            log_prob: Scalar tensor with the sum of log-probabilities for
                      the sampled tokens (suitable for REINFORCE).
        """
        logits, _ = self.forward(obj.unsqueeze(0))  # (1, L, V)
        logits = logits.squeeze(0)  # (L, V)

        dists = Categorical(logits=logits)
        tokens = dists.sample()  # (L,)
        log_prob = dists.log_prob(tokens).sum()
        return tokens, log_prob


class ListenerAgent(nn.Module):
    """Listener agent that maps objects and a message to an object choice.

    The Listener observes:
      * A set of candidate objects, each as a 3-dim integer vector.
      * A message (L-length sequence of tokens).

    It embeds objects and message, computes a representation, and scores each
    object, producing a categorical distribution over indices.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        message_length: int,
        world_color_dim: int,
        world_shape_dim: int,
        world_size_dim: int,
        config: ListenerConfig | None = None,
    ) -> None:
        super().__init__()
        self.vocab = vocab
        self.message_length = message_length
        self.config = config or ListenerConfig()

        embed_dim = self.config.hidden_dim

        # Object attribute embeddings
        self.color_emb = nn.Embedding(world_color_dim, embed_dim)
        self.shape_emb = nn.Embedding(world_shape_dim, embed_dim)
        self.size_emb = nn.Embedding(world_size_dim, embed_dim)

        # Message embedding
        self.token_emb = nn.Embedding(self.vocab.size, embed_dim)

        self.obj_encoder = nn.Sequential(
            nn.Linear(embed_dim * 3, self.config.hidden_dim),
            nn.ReLU(),
        )

        self.msg_encoder = nn.GRU(
            input_size=embed_dim,
            hidden_size=self.config.hidden_dim,
            batch_first=True,
        )

        self.final_mlp = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, 1),
        )

    def encode_objects(self, objects: torch.Tensor) -> torch.Tensor:
        """Encode objects into continuous vectors.

        Args:
            objects: Tensor of shape (num_objects, 3).

        Returns:
            Tensor of shape (num_objects, hidden_dim)
        """
        color_idx = objects[..., 0]
        shape_idx = objects[..., 1]
        size_idx = objects[..., 2]

        c = self.color_emb(color_idx)
        s = self.shape_emb(shape_idx)
        z = self.size_emb(size_idx)
        x = torch.cat([c, s, z], dim=-1)
        return self.obj_encoder(x)

    def encode_message(self, message: torch.Tensor) -> torch.Tensor:
        """Encode a message into a fixed-size vector.

        Args:
            message: Tensor of shape (L,)

        Returns:
            Tensor of shape (hidden_dim,)
        """
        emb = self.token_emb(message.unsqueeze(0))  # (1, L, E)
        _, h_n = self.msg_encoder(emb)  # h_n: (1, 1, H)
        return h_n.squeeze(0).squeeze(0)

    def forward(self, objects: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        """Compute unnormalized scores for each object.

        Args:
            objects: Tensor of shape (num_objects, 3).
            message: Tensor of shape (L,).

        Returns:
            scores: Tensor of shape (num_objects,) with unnormalized logits.
        """
        obj_repr = self.encode_objects(objects)  # (N, H)
        msg_repr = self.encode_message(message)  # (H,)

        # Broadcast message representation to each object
        msg_expand = msg_repr.unsqueeze(0).expand_as(obj_repr)  # (N, H)
        x = torch.cat([obj_repr, msg_expand], dim=-1)  # (N, 2H)

        scores = self.final_mlp(x).squeeze(-1)  # (N,)
        return scores

    def act(self, objects: torch.Tensor, message: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """Sample an object index and return its log-probability.

        Args:
            objects: Tensor of shape (num_objects, 3).
            message: Tensor of shape (L,).

        Returns:
            action: Python int index into the candidate list.
            log_prob: Scalar tensor, log-probability of the sampled action.
        """
        scores = self.forward(objects, message)  # (N,)
        dist = Categorical(logits=scores)
        idx = dist.sample()
        log_prob = dist.log_prob(idx)
        return int(idx.item()), log_prob

