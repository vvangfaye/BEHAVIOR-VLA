"""Isolated, single-chunk differentiable-policy feasibility prototype."""

from .differentiable_chunk import ChunkSample, model_actor_loss, sample_chunk_with_grad

__all__ = ["ChunkSample", "model_actor_loss", "sample_chunk_with_grad"]
