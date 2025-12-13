"""
sarm_wrapper.py

A minimal wrapper around Schrieffer/Llama-SARM-4B (or PostSAEPretrain) that:
- formats (prompt, response) using the model's chat template,
- extracts SAE latents at the pooled token,
- supports in-loop feature-level controls:
  * activation clamping/setting on selected latent dims,
  * temporary linear-head ("score") edits (optional).

The wrapper mirrors the key steps in the released SARM implementation:
- pre_process: per-token mean/std normalization
- TopkSAE.get_latents: scatter top-k values into a dense latent vector
- score: nn.Linear(latent_size -> 1) applied to SAE features
- pooling at the last (non-pad) token
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass(frozen=True)
class ClampSpec:
    """Clamp or set a specific latent dimension."""
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    set_val: Optional[float] = None


def _right_pad(seqs: List[torch.Tensor], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Right-pad a list of (1, L_i) LongTensor sequences to (B, Lmax).
    Returns (input_ids, attention_mask).
    """
    if len(seqs) == 0:
        raise ValueError("No sequences to pad.")

    seqs_1d = [s.squeeze(0) for s in seqs]
    device = seqs_1d[0].device
    lengths = torch.tensor([s.numel() for s in seqs_1d], device=device)
    max_len = int(lengths.max().item())
    B = len(seqs_1d)

    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((B, max_len), dtype=torch.long, device=device)

    for i, s in enumerate(seqs_1d):
        L = s.numel()
        input_ids[i, :L] = s
        attention_mask[i, :L] = 1

    return input_ids, attention_mask


class SARMRewardModel:
    """
    Loads SARM as a frozen reward model, but exposes pooled SAE latents so you can implement
    in-loop feature controls without training the SAE.

    Assumptions (verified in Schrieffer/Llama-SARM-4B remote code):
      - self.model has attributes: model (base transformer), sae (TopkSAE), score (Linear), sarm_use_activation (bool)
      - score returns a single scalar per sequence after pooling at the last token.
    """

    def __init__(
        self,
        model_name: str = "Schrieffer/Llama-SARM-4B",
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = torch.bfloat16,
        trust_remote_code: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            # Common Llama tokenizer setup
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Try to use flash_attention_2 if available, fall back to eager
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                device_map=self.device,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2",
            )
        except ImportError:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                device_map=self.device,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch_dtype,
                attn_implementation="eager",
            )
        self.model.eval()

        # Patch missing config.pad_token_id (seen in some repos)
        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.base = getattr(self.model, "model", None)
        self.sae = getattr(self.model, "sae", None)
        self.score_layer = getattr(self.model, "score", None)
        self.use_activation = bool(getattr(self.model, "sarm_use_activation", True))

        if self.base is None or self.sae is None or self.score_layer is None:
            raise RuntimeError(
                "Unexpected SARM model structure. Expected attributes: model, sae, score. "
                "Make sure trust_remote_code=True and you loaded a Schrieffer SARM checkpoint."
            )

    # -------------------------
    # Public API
    # -------------------------

    @torch.no_grad()
    def score(
        self,
        prompt: str,
        response: str,
        clamp: Optional[Dict[int, ClampSpec]] = None,
        head_delta: Optional[Dict[int, float]] = None,
        return_latents: bool = False,
    ) -> Union[float, Tuple[float, torch.Tensor]]:
        rewards, z = self.score_batch(
            prompts=[prompt],
            responses=[response],
            clamp=clamp,
            head_delta=head_delta,
            return_latents=True,
        )
        if return_latents:
            return float(rewards[0].item()), z[0].detach().cpu()
        return float(rewards[0].item())

    @torch.no_grad()
    def score_batch(
        self,
        prompts: Sequence[str],
        responses: Sequence[str],
        clamp: Optional[Dict[int, ClampSpec]] = None,
        head_delta: Optional[Dict[int, float]] = None,
        return_latents: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if len(prompts) != len(responses):
            raise ValueError("prompts and responses must have the same length.")
        input_ids, attention_mask = self._encode_pairs(list(prompts), list(responses))
        z_pooled = self._extract_pooled_latents(input_ids, attention_mask)
        if clamp:
            z_pooled = self._apply_clamp(z_pooled, clamp)

        rewards = self._score_with_optional_head_delta(z_pooled, head_delta=head_delta)
        if return_latents:
            return rewards, z_pooled
        return rewards

    @torch.no_grad()
    def pooled_latents(
        self,
        prompts: Sequence[str],
        responses: Sequence[str],
        clamp: Optional[Dict[int, ClampSpec]] = None,
    ) -> torch.Tensor:
        _, z = self.score_batch(prompts, responses, clamp=clamp, return_latents=True)
        return z

    # -------------------------
    # Internals
    # -------------------------

    def _encode_pairs(self, prompts: List[str], responses: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mirrors the HF model card demo (apply_chat_template + forward), but supports batching via padding.
        """
        seqs: List[torch.Tensor] = []
        for p, r in zip(prompts, responses):
            messages = [{"role": "user", "content": p}, {"role": "assistant", "content": r}]
            ids = self.tokenizer.apply_chat_template(messages, return_tensors="pt")
            seqs.append(ids.to(self.model.device))

        pad_id = int(self.model.config.pad_token_id)
        input_ids, attention_mask = _right_pad(seqs, pad_id)
        return input_ids, attention_mask

    @staticmethod
    def _pre_process(hidden_states: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        mean = hidden_states.mean(dim=-1, keepdim=True)
        std = hidden_states.std(dim=-1, keepdim=True)
        return (hidden_states - mean) / (std + eps)

    def _extract_pooled_latents(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Base transformer forward
        out = self.base(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden_states = out[0]  # [B, T, H]

        x = self._pre_process(hidden_states)
        pre_acts = self.sae.pre_acts(x)
        latents = self.sae.get_latents(pre_acts) if self.use_activation else pre_acts

        # Pool at last (non-pad) token (same as SARM forward)
        pad_id = getattr(self.model.config, "pad_token_id", None)
        B, T = input_ids.shape
        if pad_id is None:
            seq_idx = torch.full((B,), T - 1, device=input_ids.device, dtype=torch.long)
        else:
            seq_idx = (input_ids.eq(pad_id).int().argmax(dim=-1) - 1) % T

        z_pooled = latents[torch.arange(B, device=input_ids.device), seq_idx, :]  # [B, latent_size]
        return z_pooled

    @staticmethod
    def _apply_clamp(z: torch.Tensor, clamp: Dict[int, ClampSpec]) -> torch.Tensor:
        z = z.clone()
        for idx, spec in clamp.items():
            if spec.set_val is not None:
                z[:, idx] = float(spec.set_val)
            else:
                if spec.min_val is not None:
                    z[:, idx] = torch.maximum(z[:, idx], torch.tensor(spec.min_val, device=z.device, dtype=z.dtype))
                if spec.max_val is not None:
                    z[:, idx] = torch.minimum(z[:, idx], torch.tensor(spec.max_val, device=z.device, dtype=z.dtype))
        return z

    def _score_with_optional_head_delta(self, z: torch.Tensor, head_delta: Optional[Dict[int, float]]) -> torch.Tensor:
        """
        Compute score(z) with optional temporary edits to score weights.
        head_delta: {feature_idx: additive_delta_to_weight}.
        """
        if not head_delta:
            return self.score_layer(z).squeeze(-1)

        # score_layer.weight shape: [num_labels(=1), latent_size]
        weight = self.score_layer.weight
        if weight.ndim != 2 or weight.shape[0] != 1:
            raise RuntimeError(f"Unexpected score_layer weight shape: {tuple(weight.shape)}")

        # Apply deltas in-place under no_grad, then restore.
        orig = {}
        for idx, delta in head_delta.items():
            orig[idx] = weight[0, idx].detach().clone()
            weight[0, idx] = weight[0, idx] + float(delta)

        try:
            out = self.score_layer(z).squeeze(-1)
        finally:
            for idx, val in orig.items():
                weight[0, idx] = val

        return out
