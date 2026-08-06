"""
train_rag.py
============
Fine-tuning script for RAG-aware Moshi LM.

What this does
--------------
1.  Loads the pretrained LMModel (Moshi) and Mimi codec.
2.  Adds a lightweight ReferenceAdapter — a small MLP that projects
    mean-pooled reference text embeddings (from the LM's own text_emb
    table) into a [dim]-dimensional conditioning vector.  This is the
    equivalent of Kyutai's ARC encoder, built entirely from existing
    model components, requiring no external service.
3.  Applies LoRA to the main transformer and depformer inside LMModel.
4.  Enables gradient checkpointing on the transformer blocks.
5.  Freezes Mimi entirely, freezes base LM weights (only LoRA deltas
    + ReferenceAdapter train).
6.  Trains with the per-segment loss from rag_loss.py:
      - 5x weight on <ref> token frame
      - 0.5x on filler text tokens
      - 2x on body text tokens (grounding signal)
      - contrastive margin loss on body frames (wrong reference)
7.  Saves checkpoints every N steps.

ReferenceAdapter — why no external encoder
------------------------------------------
The original Moshi-RAG system uses Kyutai's ARC encoder (a remote HTTP
service) to convert reference text → [T, dim] conditioning tensors.
We cannot use ARC because:
  a) It is not open-sourced.
  b) Its output distribution is co-trained with the production model;
     injecting ARC vectors into a freshly fine-tuned model would require
     the model to already understand those vectors.

Instead we build conditioning vectors directly from the LM's own
text_emb table, which is guaranteed to be in-distribution:

    reference text
         │
    text_tokenizer.encode()
         │
    lm.text_emb(token_ids)     # [T_ref, dim] — same table the LM uses
         │
    mean_pool                  # [dim]
         │
    ReferenceAdapter (MLP)     # [dim] → [dim]  (2 layers, trained)
         │
    broadcast to [B, 1, dim]
         │
    added to embed_codes() output every frame in the body segment

The adapter has ~2M parameters for dim=4096, trains quickly, and the
LM learns to use its output through the body-segment grounding loss.

Usage
-----
    python train_rag.py \
        --train-data   data/train.jsonl \
        --eval-data    data/eval.jsonl \
        --moshi-weight /path/to/moshi.safetensors \
        --mimi-weight  /path/to/mimi.safetensors \
        --tokenizer    /path/to/tokenizer.model \
        --out-dir      checkpoints/rag_lora \
        --device       cuda

    # Resume from checkpoint
    python train_rag.py ... --resume checkpoints/rag_lora/step_1000

Hyper-parameters (matching provided config)
--------------------------------------------
    --lora-rank 128  --lora-scaling 2.0
    --lr 2e-6  --weight-decay 0.1
    --batch-size 16  --max-steps 2000
    --duration-sec 100
    --first-codebook-weight 100.0  --text-padding-weight 0.5
    --ref-token-weight 5.0  --body-weight 2.0  --filler-weight 0.5
    --contrastive-weight 0.3  --contrastive-margin 0.3
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
from functools import partial

import numpy as np
import sentencepiece
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from huggingface_hub import hf_hub_download

# Project imports — same package as offline inference
from .models import loaders
from .models.lm import LMModel

# Our pipeline modules
from .rag_loss_new import compute_rag_losses, SegmentMeta, make_segment_weight_tensor
from .dataset import build_data_loader
from .interleaver import InterleavedTokenizer, Interleaver, Batch

logger = logging.getLogger(__name__)

from .models.lm import ScaledEmbedding  
def forward_train_checkpointed(self: LMModel, codes: torch.Tensor):
    from .models.lm import _delay_sequence, _undelay_sequence, LMOutput
    from torch.utils.checkpoint import checkpoint

    B, K, T = codes.shape
    initial = self._get_initial_token().expand(B, -1, -1)
    delayed = _delay_sequence(self.delays, codes, initial)
    delayed = torch.cat([initial, delayed], dim=2)

    def _run_transformer(seq):
        return self.forward_codes(seq)
    transformer_out, text_logits = checkpoint(_run_transformer, delayed[:, :, :-1], use_reentrant=False)

    def _run_depformer(delayed_slice, t_out):
        return self.forward_depformer_training(delayed_slice, t_out)
    logits = checkpoint(_run_depformer, delayed[:, :, 1:], transformer_out, use_reentrant=False)

    logits, logits_mask = _undelay_sequence(
        self.delays[self.audio_offset: self.audio_offset + self.dep_q],
        logits, fill_value=float("NaN"),
    )
    logits_mask &= (codes[:, self.audio_offset: self.audio_offset + self.dep_q] != self.zero_token_id)
    text_logits, text_logits_mask = _undelay_sequence(self.delays[:1], text_logits, fill_value=float("NaN"))
    text_logits_mask &= codes[:, :1] != self.zero_token_id
    return LMOutput(logits, logits_mask, text_logits, text_logits_mask)

class ReferenceBuffer:
    """Holds recently-seen RAG reference strings so wrong-reference sampling
    for the contrastive loss isn't limited to the current (tiny) micro-batch."""
    def __init__(self, maxlen: int = 128):
        from collections import deque
        self.buf: deque[str] = deque(maxlen=maxlen)

    def add(self, refs: list[str]):
        self.buf.extend(r for r in refs if r)

    def sample_wrong(self, references: list[str], seg_metas, rng: random.Random) -> list[str]:
        wrong = [""] * len(references)
        pool = list(self.buf)
        for i in range(len(references)):
            if not references[i]:
                continue
            candidates = [r for r in pool if r != references[i]]
            if candidates:
                wrong[i] = rng.choice(candidates)
        return wrong

def _safe_id_to_piece(tokenizer, i: int, lm_model=None) -> str:
    """id_to_piece raises IndexError on sentinel ids that aren't real
    SentencePiece pieces (zero_token_id=-1, text_initial_token_id, etc).
    Diagnostic-only — never used for anything that affects training."""
    if i < 0:
        return f"<ZERO:{i}>"
    if lm_model is not None and i == lm_model.text_initial_token_id:
        return "<TEXT_INIT>"
    if i >= tokenizer.GetPieceSize():
        return f"<OOV:{i}>"
    return tokenizer.id_to_piece(i)
    
_lookup_diag_state = {"correct": 0, "total": 0}
def print_lookup_window(codes, seg_metas, tokenizer, step, window=6, max_print=2, lm_model=None):
    printed = 0
    T = codes.shape[-1]
    for meta in seg_metas:
        for span in meta.spans_of("lookup"):
            if printed >= max_print:
                return
            b = meta.batch_idx
            s = max(0, span.start_frame - window)
            e = min(T, span.start_frame + window)
            ids = codes[b, 0, s:e].tolist()
            pieces = [_safe_id_to_piece(tokenizer, i, lm_model) for i in ids]   # CHANGED
            marked = [f"[{p}]" if (s + j) == span.start_frame else p for j, p in enumerate(pieces)]
            logger.info(f"[diag-window] step {step} b={b} lookup_start={span.start_frame} "
                        f"window[{s}:{e}]={' '.join(marked)}")
            printed += 1        
def add_special_tokens_and_resize(
    lm_base: LMModel,
    tokenizer: sentencepiece.SentencePieceProcessor,
    new_tokens: tuple[str, ...] = ("<ref_start>", "<ref_end>", "<lookup>"),
) -> dict[str, int]:
    old_text_card = lm_base.text_card
    n_new = len(new_tokens)

    new_ids = []
    for i, tok in enumerate(new_tokens):
        tid = tokenizer.piece_to_id(tok)
        if tid == tokenizer.unk_id():
            raise ValueError(f"{tok!r} not in tokenizer vocab — run add_ref_tokens_to_spm.py first.")
        expected = old_text_card + i
        if tid != expected:
            raise ValueError(f"{tok!r} has id {tid}, expected {expected} (must be contiguous, appended last).")
        new_ids.append(tid)

    device = lm_base.text_emb.weight.device
    dtype = lm_base.text_emb.weight.dtype
    dim = lm_base.text_emb.embedding_dim
    new_text_card = old_text_card + n_new

    # --- text_emb (main transformer input) --- [unchanged from before]
    old_emb = lm_base.text_emb
    old_pad_row_emb = old_emb.weight.data[old_text_card].clone()
    norm_emb = old_emb.norm is not None

    new_emb = ScaledEmbedding(
        new_text_card + 1, dim, norm=norm_emb, zero_idx=lm_base.zero_token_id,
        device=device, dtype=dtype,
    )
    with torch.no_grad():
        new_emb.weight.data[:old_text_card] = old_emb.weight.data[:old_text_card]
        centroid = old_emb.weight.data[:old_text_card].mean(dim=0)
        for tid in new_ids:
            new_emb.weight.data[tid] = centroid + 0.02 * torch.randn_like(centroid)
        new_emb.weight.data[new_text_card] = old_pad_row_emb

    # --- text_linear (output projection) --- [unchanged from before]
    old_linear = lm_base.text_linear
    extra_text = 1 if lm_base.existing_text_padding_id is None else 0
    old_pad_row_out = old_linear.weight.data[old_text_card].clone() if extra_text else None
    old_pad_bias = (
        old_linear.bias.data[old_text_card].clone()
        if (extra_text and old_linear.bias is not None) else None
    )
    new_out_dim = new_text_card + extra_text
    new_linear = torch.nn.Linear(dim, new_out_dim, bias=old_linear.bias is not None, device=device, dtype=dtype)
    with torch.no_grad():
        new_linear.weight.data[:old_text_card] = old_linear.weight.data[:old_text_card]
        out_centroid = old_linear.weight.data[:old_text_card].mean(dim=0)
        for tid in new_ids:
            new_linear.weight.data[tid] = out_centroid + 0.02 * torch.randn_like(out_centroid)
        if extra_text:
            new_linear.weight.data[new_text_card] = old_pad_row_out
        if new_linear.bias is not None:
            new_linear.bias.data[:old_text_card] = old_linear.bias.data[:old_text_card]
            for tid in new_ids:
                new_linear.bias.data[tid] = 0.0
            if extra_text and old_pad_bias is not None:
                new_linear.bias.data[new_text_card] = old_pad_bias

    # --- NEW: depformer_text_emb — the sibling table forward_depformer_training
    # reads sequence[:, 0] (text codebook) against. Same ids, same resize
    # pattern, but its own dim (depformer_dim) and its own padding/initial
    # row semantics. This is what was missing and caused the OOB lookup.
    old_dep_emb = lm_base.depformer_text_emb
    dep_dim = old_dep_emb.embedding_dim
    dep_norm = old_dep_emb.norm is not None
    old_dep_pad_row = old_dep_emb.weight.data[old_text_card].clone()

    new_dep_emb = ScaledEmbedding(
        new_text_card + 1, dep_dim, norm=dep_norm, zero_idx=lm_base.zero_token_id,
        device=old_dep_emb.weight.device, dtype=old_dep_emb.weight.dtype,
    )
    with torch.no_grad():
        new_dep_emb.weight.data[:old_text_card] = old_dep_emb.weight.data[:old_text_card]
        dep_centroid = old_dep_emb.weight.data[:old_text_card].mean(dim=0)
        for tid in new_ids:
            new_dep_emb.weight.data[tid] = dep_centroid + 0.02 * torch.randn_like(dep_centroid)
        new_dep_emb.weight.data[new_text_card] = old_dep_pad_row

    lm_base.text_emb = new_emb
    lm_base.text_linear = new_linear
    lm_base.depformer_text_emb = new_dep_emb      # NEW
    lm_base.text_card = new_text_card

    # --- gradient masks: freeze everything except the new rows, in ALL THREE tables ---
    emb_mask = torch.zeros(new_text_card + 1, 1, device=device, dtype=dtype)
    emb_mask[new_ids] = 1.0
    new_emb.weight.register_hook(lambda g, m=emb_mask: g * m)

    lin_mask = torch.zeros(new_out_dim, 1, device=device, dtype=dtype)
    lin_mask[new_ids] = 1.0
    new_linear.weight.register_hook(lambda g, m=lin_mask: g * m)
    if new_linear.bias is not None:
        lin_bias_mask = torch.zeros(new_out_dim, device=device, dtype=dtype)
        lin_bias_mask[new_ids] = 1.0
        new_linear.bias.register_hook(lambda g, m=lin_bias_mask: g * m)

    dep_emb_mask = torch.zeros(new_text_card + 1, 1, device=old_dep_emb.weight.device, dtype=old_dep_emb.weight.dtype)
    dep_emb_mask[new_ids] = 1.0
    new_dep_emb.weight.register_hook(lambda g, m=dep_emb_mask: g * m)    # NEW

    logger.info(
        f"resized text vocab {old_text_card} -> {new_text_card} "
        f"(+{n_new}: {dict(zip(new_tokens, new_ids))}); padding/initial row moved "
        f"{old_text_card} -> {new_text_card}; also resized depformer_text_emb"
    )
    return dict(zip(new_tokens, new_ids))
# ---------------------------------------------------------------------------
# Reference adapter — replaces ARC encoder
# ---------------------------------------------------------------------------

class RefResampler(nn.Module):
    """Encodes a batch of raw reference strings into a fixed n_queries
    vectors each, via cross-attention pooling (Perceiver-style). Frozen
    token embedding lookup (shares the base model's vocab/embedding
    space) + a small trainable cross-attention stack. This is the module
    that actually carries 'read the reference' signal now — sequence
    length is decoupled from reference length entirely.
    """
    def __init__(self, base_text_emb: nn.Embedding, d_model: int,
                 n_queries: int = 16, n_layers: int = 2, n_heads: int = 8):
        super().__init__()
        object.__setattr__(self, "base_text_emb", base_text_emb)
        self.n_queries = n_queries
        self.queries = nn.Parameter(torch.randn(n_queries, d_model) * 0.02)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attn": nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                "norm1": nn.LayerNorm(d_model),
                "ff": nn.Sequential(
                    nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model),
                ),
                "norm2": nn.LayerNorm(d_model),
            })
            for _ in range(n_layers)
        ])
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, ref_ids: torch.Tensor, ref_mask: torch.Tensor) -> torch.Tensor:
        """ref_ids: [B, L] padded token ids. ref_mask: [B, L] bool, True=real token.
        Returns: [B, n_queries, D]."""
        with torch.no_grad():
            kv = self.base_text_emb(ref_ids)               # [B, L, D] — frozen lookup
        B = ref_ids.shape[0]
        q = self.queries.unsqueeze(0).expand(B, -1, -1)     # [B, K, D]
        key_padding_mask = ~ref_mask                        # True = ignore
        for layer in self.layers:
            attn_out, _ = layer["attn"](q, kv, kv, key_padding_mask=key_padding_mask)
            q = layer["norm1"](q + attn_out)
            q = layer["norm2"](q + layer["ff"](q))
        return self.out_norm(q)

class RefInjectingEmbedding(nn.Module):
    """Drop-in replacement for lm_base.text_emb (a ScaledEmbedding).
    embed_codes() calls self.text_emb(input_sequence[:, 0]) directly, so
    swapping the attribute is enough — no monkeypatching of embed_codes
    itself needed. Normal ids pass straight through to the wrapped
    ScaledEmbedding (zero_idx handling included); ref_slot positions get
    overwritten with the resampler's continuous output for that sample."""
    def __init__(self, base_text_emb):
        super().__init__()
        self.base = base_text_emb
        self._pending: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    @property
    def embedding_dim(self):
        return self.base.embedding_dim

    @property
    def weight(self):
        return self.base.weight

    def set_injection(self, batch_idx: torch.Tensor, frame_idx: torch.Tensor, vectors: torch.Tensor):
        self._pending = (batch_idx, frame_idx, vectors)

    def clear_injection(self):
        self._pending = None

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        emb = self.base(ids)
        if self._pending is not None:
            b_idx, t_idx, vecs = self._pending
            emb = emb.clone()
            emb[b_idx, t_idx] = vecs.to(emb.dtype)
        return emb
# ---------------------------------------------------------------------------
# LoRA application
# ---------------------------------------------------------------------------

def apply_lora(lm, rank=128, lora_alpha=256.0, target_modules=None):
    from peft import LoraConfig, get_peft_model, TaskType
    if target_modules is None:
        target_modules = ["in_proj", "out_proj", "fc1", "fc2", "linear", "proj"]

    config = LoraConfig(
        r=rank, lora_alpha=lora_alpha, target_modules=target_modules,
        lora_dropout=0.05, bias="none", task_type=TaskType.FEATURE_EXTRACTION,
        modules_to_save=None,
    )
    lm = get_peft_model(lm, config)

    # peft's suffix matching can accidentally net text_linear (e.g.
    # "text_linear".endswith("linear") is True). text_emb/text_linear are
    # handled entirely by add_special_tokens_and_resize's row-masked
    # gradients, never by LoRA — unwrap if peft grabbed them.
    base = lm.base_model.model if hasattr(lm.base_model, "model") else lm.base_model
    for name in ("text_emb", "text_linear", "depformer_text_emb"):
        mod = getattr(base, name, None)
        if mod is not None and hasattr(mod, "base_layer"):
            logger.warning(f"peft wrapped {name} in LoRA despite exclusion intent — unwrapping")
            setattr(base, name, mod.base_layer)

    lm.print_trainable_parameters()
    return lm


# ---------------------------------------------------------------------------
# Gradient checkpointing
# ---------------------------------------------------------------------------

def enable_gradient_checkpointing(lm: nn.Module) -> None:
    """
    Enable gradient checkpointing on the transformer layers inside LMModel.

    peft's get_peft_model exposes enable_input_require_grads which is needed
    for gradient checkpointing to work when inputs don't have requires_grad.
    Then we enable checkpointing on the underlying transformer modules.
    """
    # peft wrapper
    if hasattr(lm, "enable_input_require_grads"):
        lm.enable_input_require_grads()

    # Enable on the two StreamingTransformer modules
    base = lm.base_model if hasattr(lm, "base_model") else lm
    for name in ("transformer", "depformer"):
        module = getattr(base, name, None)
        if module is not None and hasattr(module, "gradient_checkpointing_enable"):
            module.gradient_checkpointing_enable()
            logger.info(f"gradient checkpointing enabled on lm.{name}")
        elif module is not None:
            # StreamingTransformer may not expose the method directly;
            # wrap its layers manually
            if hasattr(module, "layers"):
                module.gradient_checkpointing = True
                logger.info(f"set gradient_checkpointing=True on lm.{name}")


# ---------------------------------------------------------------------------
# Streaming sum injection — equivalent of apply_pending_streaming_sum_condition
# ---------------------------------------------------------------------------

def inject_streaming_sum(
    embed_codes_output: torch.Tensor,   # [B, T, dim]
    streaming_sum: torch.Tensor,        # [B, dim]
    seg_metas: list[SegmentMeta | None],
    T: int,
) -> torch.Tensor:
    """
    Add the reference conditioning vector to embed_codes output at body frames.

    streaming_sum[b] is added to every frame from body_start onward for batch
    element b.  Non-RAG elements (seg_meta=None) or elements with zero
    streaming_sum (reference='') are unaffected.

    This is the forward-pass equivalent of:
        apply_pending_streaming_sum_condition()  (System 2, BatchRunner)

    Shape: embed_codes_output [B, T, dim] is modified in-place and returned.
    """
    B, _, dim = embed_codes_output.shape
    for b, meta in enumerate(seg_metas):
        if meta is None:
            continue
        body_start = meta.body_start
        if body_start >= T:
            continue
        # streaming_sum[b]: [dim] → broadcast to [T-body_start, dim]
        cond = streaming_sum[b].unsqueeze(0)   # [1, dim]
        embed_codes_output[b, body_start:] = (
            embed_codes_output[b, body_start:] + cond
        )
    return embed_codes_output

def embed_codes_split(lm: LMModel, sequence: torch.Tensor):
    """Same math as LMModel.embed_codes, but keeps audio-sum and text_emb separate."""
    audio_sum = None
    for cb_index in range(lm.num_audio_codebooks):
        audio_emb = lm.emb[cb_index](sequence[:, cb_index + lm.audio_offset])
        audio_sum = audio_emb if audio_sum is None else audio_sum + audio_emb
    text_emb = lm.text_emb(sequence[:, 0])
    return audio_sum, text_emb


def inject_streaming_sum_text_channel(text_emb, streaming_sum, seg_metas, T):
    for b, meta in enumerate(seg_metas):
        if meta is None:
            continue
        if meta.body_start >= T:
            continue
        text_emb[b, meta.body_start:] = text_emb[b, meta.body_start:] + streaming_sum[b].unsqueeze(0)
    return text_emb
# ---------------------------------------------------------------------------
# Patched forward pass
# We monkey-patch LMModel.forward_train to accept streaming_sum
# without modifying the original model source.
# ---------------------------------------------------------------------------

def forward_train_with_streaming_sum(
    self: LMModel,
    codes: torch.Tensor,
    streaming_sum: torch.Tensor | None = None,
    seg_metas: list[SegmentMeta | None] | None = None,
):
    """
    Drop-in replacement for LMModel.forward_train that injects streaming_sum.

    Identical to the original except embed_codes() output is modified by
    inject_streaming_sum() before being passed to forward_embeddings().

    Attached to LMModel instances at training time; not persistent.
    """
    from .models.lm import _delay_sequence, _undelay_sequence

    B, K, T = codes.shape
    initial  = self._get_initial_token().expand(B, -1, -1)
    delayed  = _delay_sequence(self.delays, codes, initial)
    delayed  = torch.cat([initial, delayed], dim=2)

    audio_sum, text_emb = embed_codes_split(self, delayed[:, :, :-1])
    if streaming_sum is not None and seg_metas is not None:
        text_emb = inject_streaming_sum_text_channel(text_emb, streaming_sum, seg_metas, T)
                                                    
    embedded = audio_sum + text_emb

    # --- forward through transformer + depformer (same as original) ---
    transformer_out, text_logits = self.forward_embeddings(embedded)
    logits = self.forward_depformer_training(delayed[:, :, 1:], transformer_out)

    logits, logits_mask = _undelay_sequence(
        self.delays[self.audio_offset : self.audio_offset + self.dep_q],
        logits,
        fill_value=float("NaN"),
    )
    logits_mask &= (
        codes[:, self.audio_offset : self.audio_offset + self.dep_q]
        != self.zero_token_id
    )
    text_logits, text_logits_mask = _undelay_sequence(
        self.delays[:1], text_logits, fill_value=float("NaN")
    )
    text_logits_mask &= codes[:, :1] != self.zero_token_id

    from .models.lm import LMOutput
    return LMOutput(logits, logits_mask, text_logits, text_logits_mask)


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------

# save_checkpoint / load_checkpoint signatures — take resampler, use resampler
def save_checkpoint(out_dir: Path, step: int, lm: nn.Module, resampler: RefResampler,
                     optimizer: torch.optim.Optimizer, scheduler, stats: dict) -> None:
    ckpt_dir = out_dir / f"step_{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(lm, "save_pretrained"):
        lm.save_pretrained(ckpt_dir / "lora")
    else:
        torch.save(lm.state_dict(), ckpt_dir / "lm_state.pt")
    torch.save(resampler.state_dict(), ckpt_dir / "resampler.pt")

    base_lm = lm.base_model.model if hasattr(lm.base_model, "model") else lm.base_model
    torch.save({
        "text_emb": base_lm.text_emb.state_dict(),
        "text_linear": base_lm.text_linear.state_dict(),
        "depformer_text_emb": base_lm.depformer_text_emb.state_dict(),
    }, ckpt_dir / "resized_embeddings.pt")

    torch.save({
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "step": step, "stats": stats,
    }, ckpt_dir / "optim.pt")
    logger.info(f"[ckpt] saved step {step} → {ckpt_dir}")


def load_checkpoint(ckpt_dir: Path, lm: nn.Module, resampler: RefResampler,
                     optimizer: torch.optim.Optimizer, scheduler) -> int:
    lora_path = ckpt_dir / "lora"
    if lora_path.exists() and hasattr(lm, "load_adapter"):
        lm.load_adapter(str(lora_path), adapter_name="default")
        logger.info(f"[ckpt] loaded LoRA from {lora_path}")
    elif (ckpt_dir / "lm_state.pt").exists():
        lm.load_state_dict(torch.load(ckpt_dir / "lm_state.pt"))

    resampler_path = ckpt_dir / "resampler.pt"
    if resampler_path.exists():
        resampler.load_state_dict(torch.load(resampler_path))
        logger.info(f"[ckpt] loaded resampler from {resampler_path}")

    optim_path = ckpt_dir / "optim.pt"
    if optim_path.exists():
        state = torch.load(optim_path)
        optimizer.load_state_dict(state["optimizer"])
        if scheduler and state.get("scheduler"):
            scheduler.load_state_dict(state["scheduler"])
        step = state["step"]
        logger.info(f"[ckpt] resumed from step {step}")
        return step
    return 0


# ---------------------------------------------------------------------------
# Wrong-reference construction for contrastive loss
# ---------------------------------------------------------------------------

def build_wrong_references(
    references: list[str],
    seg_metas: list[SegmentMeta | None],
    rng: random.Random,
) -> list[str]:
    """
    For each RAG slot, swap its reference with a random other slot's reference.
    Non-RAG slots (meta=None or ref='') get empty string.

    The contrastive loss only fires on body frames, so wrong refs for non-RAG
    slots produce zero gradient regardless.
    """
    rag_refs = [
        (i, references[i])
        for i, m in enumerate(seg_metas)
        if m is not None and references[i]
    ]
    if len(rag_refs) < 2:
        # Can't swap with only one RAG example — return empty (skip contrastive)
        return [""] * len(references)

    wrong = list(references)
    for i, ref in rag_refs:
        # Pick a different RAG slot's reference
        others = [(j, r) for j, r in rag_refs if j != i]
        j, other_ref = rng.choice(others)
        wrong[i] = other_ref
    return wrong


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def _build_ref_injection(batch, seg_metas, tokenizer, resampler, device, ref_buffer, rng, args):
    """Tokenizes each sample's reference text (capped), runs the resampler,
    and returns flattened (batch_idx, frame_idx, vectors) targets for
    RefInjectingEmbedding. Samples with no ref span are skipped entirely —
    zero resampler cost for non-RAG turns."""
    ref_texts, slot_ranges, batch_positions = [], [], []
    for i, meta in enumerate(seg_metas):
        if meta is None:
            continue
        slots = meta.spans_of("ref_slot")
        if not slots:
            continue
        ref_texts.append(batch.reference_texts[meta.batch_idx] or "")
        slot_ranges.append((slots[0].start_frame, slots[0].end_frame))
        batch_positions.append(meta.batch_idx)

    if not ref_texts:
        return None

    ref_buffer.add(ref_texts)  # keep pool fresh regardless of wrong-ref usage this step

    max_len = args.max_ref_source_tokens
    tok_ids = torch.full((len(ref_texts), max_len), tokenizer.pad_id() if tokenizer.pad_id() >= 0 else 0,
                          dtype=torch.long)
    mask = torch.zeros((len(ref_texts), max_len), dtype=torch.bool)
    for i, text in enumerate(ref_texts):
        ids = tokenizer.encode(text)[:max_len]
        tok_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        mask[i, :len(ids)] = True

    tok_ids, mask = tok_ids.to(device), mask.to(device)
    vecs = resampler(tok_ids, mask)   # [N, K, D]

    b_idx, t_idx, flat_vecs = [], [], []
    for i, (b, (s, e)) in enumerate(zip(batch_positions, slot_ranges)):
        k = e - s
        b_idx.extend([b] * k)
        t_idx.extend(range(s, e))
        flat_vecs.append(vecs[i, :k])

    b_idx = torch.tensor(b_idx, device=device, dtype=torch.long)
    t_idx = torch.tensor(t_idx, device=device, dtype=torch.long)
    flat_vecs = torch.cat(flat_vecs, dim=0)
    return b_idx, t_idx, flat_vecs


def training_step(batch, lm, resampler, ref_injector, ref_buffer, args, rng, device, tokenizer, step=0, diag_every=20):
    codes = batch.codes.to(device)
    seg_metas = batch.segment_metas
    base_lm = lm.base_model if hasattr(lm, "base_model") else lm

    injection = _build_ref_injection(batch, seg_metas, tokenizer, resampler, device, ref_buffer, rng, args)
    if injection is not None:
        ref_injector.set_injection(*injection)

    try:
        output = forward_train_checkpointed(base_lm, codes=codes)

        filtered_metas = [m for m in seg_metas if m is not None]
        if step % diag_every == 0:
            print_lookup_window(codes=codes, seg_metas=filtered_metas, tokenizer=tokenizer, step=step)

        text_loss, audio_loss = compute_rag_losses(
            model=base_lm, output=output, codes=codes, seg_metas=filtered_metas, args=args,
        )
        loss = text_loss + audio_loss
        loss.backward()
    finally:
        ref_injector.clear_injection()   # always clear, even on exception

    return {"loss": loss.item(), "text_loss": text_loss.item(), "audio_loss": audio_loss.item()}


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    device = torch.device(args.device)

    # ── Load models ───────────────────────────────────────────────────────
    logger.info("loading tokenizer")
    if args.tokenizer is None:
        args.tokenizer = hf_hub_download(args.hf_repo, loaders.TEXT_TOKENIZER_NAME)
    tokenizer = sentencepiece.SentencePieceProcessor(args.tokenizer)

    logger.info("loading mimi (frozen)")
    if args.mimi_weight is None:
        args.mimi_weight = hf_hub_download(args.hf_repo, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(args.mimi_weight, device)
    mimi.eval()
    for p in mimi.parameters():
        p.requires_grad_(False)   # Mimi never trains

    logger.info("loading moshi LM")
    if args.moshi_weight is None:
        args.moshi_weight = hf_hub_download(args.hf_repo, loaders.MOSHI_NAME)
    lm_base: LMModel = loaders.get_moshi_lm(args.moshi_weight, device=device, cpu_offload=False)
    lm_base.eval()
    model_dtype = next(lm_base.parameters()).dtype
    lm_base = lm_base.to(dtype=model_dtype)

    # 1. LoRA FIRST — freezes everything, adds adapters to transformer/depformer.
    lm = apply_lora(lm_base, rank=args.lora_rank, lora_alpha=args.lora_rank * args.lora_scaling)

    # 2. THEN resize text vocab — new tensors are untouched by peft's freeze pass.
    new_token_ids = add_special_tokens_and_resize(
        lm_base, tokenizer, new_tokens=("<ref_start>", "<ref_end>", "<lookup>")
    )

    # 3. THEN wrap the (resized) text_emb for resampler injection.
    d_model = lm_base.text_emb.embedding_dim
    resampler = RefResampler(
        base_text_emb=lm_base.text_emb, d_model=d_model, n_queries=args.n_ref_slots,
    ).to(device=device, dtype=model_dtype)

    ref_injector = RefInjectingEmbedding(lm_base.text_emb)
    lm_base.text_emb = ref_injector

    if args.gradient_checkpointing:
        enable_gradient_checkpointing(lm)
    lm.train()
    resampler.train()

    trainable_params = [p for p in lm.parameters() if p.requires_grad]
    trainable_params += [p for p in resampler.parameters() if p.requires_grad]
    logger.info(f"trainable (LoRA+resampler): {sum(p.numel() for p in trainable_params):,} params")

    new_text_params = (
        list(lm_base.text_emb.parameters())
        + list(lm_base.text_linear.parameters())
        + list(lm_base.depformer_text_emb.parameters()) 
    )
    new_text_param_ids = {id(p) for p in new_text_params}
    lora_params = [p for p in lm.parameters() if p.requires_grad and id(p) not in new_text_param_ids]

    optimizer = AdamW([
        {"params": lora_params, "lr": args.lr, "weight_decay": args.weight_decay},
        {"params": new_text_params, "lr": args.lr, "weight_decay": 0.0},
        {"params": [p for p in resampler.parameters() if p.requires_grad],
         "lr": args.resampler_lr, "weight_decay": args.weight_decay},
    ], betas=(0.9, 0.95), eps=1e-8)

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.max_steps,
        eta_min=args.lr * 0.1,
    )

    # ── Dataset ───────────────────────────────────────────────────────────
    interleaver = Interleaver(
        tokenizer=tokenizer,
        audio_frame_rate=mimi.frame_rate,
        text_padding=lm_base.base_model.text_padding_token_id
            if hasattr(lm_base, "base_model") else lm_base.text_padding_token_id,
        end_of_text_padding=lm_base.base_model.end_of_text_padding_id
            if hasattr(lm_base, "base_model") else lm_base.end_of_text_padding_id,
        zero_padding=lm_base.base_model.zero_token_id
            if hasattr(lm_base, "base_model") else lm_base.zero_token_id,
        keep_main_only=False,
        use_bos_eos=True,
        main_speaker_label="MODEL",
        device=str(device),
    )
    instruct_tokenizer = InterleavedTokenizer(
        mimi=mimi, interleaver=interleaver, duration_sec=args.duration_sec,
        n_ref_slots=args.n_ref_slots,
        max_ref_source_tokens=args.max_ref_source_tokens,
    )

    train_loader = build_data_loader(
        instruct_tokenizer=instruct_tokenizer,
        args=args,
        batch_size=args.batch_size,
        seed=42,
        rank=0,
        world_size=1,
        is_eval=False,
    )
    eval_loader = build_data_loader(
        instruct_tokenizer=instruct_tokenizer,
        args=args,
        batch_size=args.batch_size,
        seed=0,
        rank=0,
        world_size=1,
        is_eval=True,
    ) if args.eval_data else None

    # ── Resume ────────────────────────────────────────────────────────────
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(Path(args.resume), lm, resampler, optimizer, scheduler)

    # ── Training loop ─────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(42)
    step = start_step
    log_interval  = args.log_every
    eval_interval = args.eval_every
    save_interval = args.save_every

    running = {k: 0.0 for k in ("loss", "text_loss", "audio_loss", "contrastive")}
    running_n = 0
    t0 = time.time()

    logger.info(f"starting training from step {step}")

    optimizer.zero_grad()
    accum_steps = 0

    ref_buffer = ReferenceBuffer()  

    for batch in train_loader:
        if step >= args.max_steps:
            break

        scalars = training_step(
            batch=batch, lm=lm, resampler=resampler, ref_injector=ref_injector,
            ref_buffer=ref_buffer, args=args, rng=rng, device=device,
            tokenizer=tokenizer, step=step,
        )
        accum_steps += 1
        for k, v in scalars.items():
            running[k] += v
        running_n += 1

        if accum_steps >= args.grad_accum:
            torch.nn.utils.clip_grad_norm_(
                list(lm.parameters()) + list(resampler.parameters()), max_norm=1.0,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            accum_steps = 0

            # ── Logging ───────────────────────────────────────────────────
            if step % log_interval == 0:
                elapsed = time.time() - t0
                avg     = {k: running[k] / max(running_n, 1) for k in running}
                lr      = scheduler.get_last_lr()[0]
                logger.info(
                    f"step {step:5d}/{args.max_steps} "
                    f"loss={avg['loss']:.4f} "
                    f"text={avg['text_loss']:.4f} "
                    f"audio={avg['audio_loss']:.4f} "
                    f"contrastive={avg['contrastive']:.4f} "
                    f"lr={lr:.2e} "
                    f"elapsed={elapsed:.0f}s"
                )
                running   = {k: 0.0 for k in running}
                running_n = 0

            # ── Eval ──────────────────────────────────────────────────────
            if eval_loader is not None and step % eval_interval == 0:
                lm.eval()
                resampler.eval()
                eval_losses = {k: 0.0 for k in ("loss", "text_loss", "audio_loss", "contrastive")}
                eval_n = 0
                with torch.no_grad():
                    for eval_batch in eval_loader:
                        s = training_step(
                            batch=eval_batch, lm=lm, resampler=resampler, ref_injector=ref_injector,
                            ref_buffer=ref_buffer, args=args, rng=rng, device=device,
                            tokenizer=tokenizer, step=step,
                        )
                        for k in eval_losses:
                            eval_losses[k] += s.get(k, 0.0)
                        eval_n += 1
                        if eval_n >= 50:
                            break
                avg_eval = {k: v / max(eval_n, 1) for k, v in eval_losses.items()}
                logger.info(f"[eval] step {step} loss={avg_eval['loss']:.4f} "
                            f"text={avg_eval['text_loss']:.4f} audio={avg_eval['audio_loss']:.4f} "
                            f"contrastive={avg_eval['contrastive']:.4f}")
                lm.train()
                resampler.train()

            if step % save_interval == 0:
                save_checkpoint(out_dir, step, lm, resampler, optimizer, scheduler, stats={"step": step})

    save_checkpoint(out_dir, step, lm, resampler, optimizer, scheduler, stats={"step": step, "final": True})
    
    logger.info(f"training complete at step {step}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data
    ap.add_argument("--train-data",  required=True, help="train.jsonl manifest path")
    ap.add_argument("--eval-data",   default="",    help="eval.jsonl manifest path (optional)")
    ap.add_argument("--duration-sec", type=float, default=100.0)
    ap.add_argument("--batch-size",   type=int,   default=16)

    # Model
    ap.add_argument("--hf-repo",      default=loaders.DEFAULT_REPO)
    ap.add_argument("--moshi-weight", default=None)
    ap.add_argument("--mimi-weight",  default=None)
    ap.add_argument("--tokenizer",    default=None)
    ap.add_argument("--device",       default="cuda")

    # LoRA
    ap.add_argument("--lora-rank",    type=int,   default=128)
    ap.add_argument("--lora-scaling", type=float, default=2.0)
    ap.add_argument("--ft-embed",     action="store_true",
                    help="Also fine-tune text_emb (default: frozen)")
    ap.add_argument("--gradient-checkpointing", action="store_true", default=True)

    ap.add_argument("--n-ref-slots", type=int, default=16,
                     help="Fixed number of resampler output vectors injected per reference (replaces --max-ref-tokens)")
    ap.add_argument("--max-ref-source-tokens", type=int, default=400,
                     help="Cap on raw reference tokens the resampler reads; does NOT affect main sequence length")
    ap.add_argument("--resampler-lr", type=float, default=1e-4,
                     help="Consider a separate, higher LR for the resampler vs LoRA — it's trained from scratch")

    # Optimiser
    ap.add_argument("--lr",           type=float, default=2e-6)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--max-steps",    type=int,   default=2000)
    ap.add_argument("--grad-accum",   type=int,   default=1,
                    help="Gradient accumulation steps (default: 1)")

    # Loss weights
    ap.add_argument("--first-codebook-weight", dest="first_codebook_weight_multiplier",
                    type=float, default=100.0)
    ap.add_argument("--text-padding-weight",   type=float, default=0.5)
    ap.add_argument("--ref-token-weight",      type=float, default=5.0)
    ap.add_argument("--lead-weight",           type=float, default=1.0)
    ap.add_argument("--filler-weight",         type=float, default=0.5)
    ap.add_argument("--body-weight",           type=float, default=2.0)
    ap.add_argument("--no-rag-weight",         type=float, default=1.0)
    ap.add_argument("--contrastive-weight",    type=float, default=0.3)
    ap.add_argument("--contrastive-margin",    type=float, default=0.3)

    # Checkpointing / logging
    ap.add_argument("--out-dir",    default="checkpoints/rag_lora")
    ap.add_argument("--resume",     default=None,
                    help="Path to checkpoint directory to resume from")
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--log-every",  type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=200)

    # DataArgs compatibility (required by build_data_loader)
    ap.add_argument("--shuffle", action="store_true", default=True)

    args = ap.parse_args()

    # Expose as DataArgs-compatible attributes
    args.train_data = args.train_data
    args.eval_data  = args.eval_data

    with torch.no_grad():
        pass   # not wrapping train() — we need gradients

    train(args)


if __name__ == "__main__":
    main()