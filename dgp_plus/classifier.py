"""LLM + LoRA classifier with first-token readout (paper Eq. 13-14).

The two equations normalise differently, as the paper writes them:
    training   cross-entropy over the FULL vocabulary at the first generated position, target "Yes" or "No"
    inference  softmax over ONLY the Yes and No logits

No text is ever generated for classification. Label token ids are resolved at run time, and the run stops if "Yes" or
"No" is not one token right after the rendered prompt, rather than silently scoring the wrong token.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LabelTokens:
    yes_id: int
    no_id: int


def first_token_loss(logits: torch.Tensor, labels: torch.Tensor, tokens: LabelTokens) -> torch.Tensor:
    """Eq. 13. labels: 1 fraud, 0 benign."""
    targets = torch.where(labels.bool(), torch.full_like(labels, tokens.yes_id), torch.full_like(labels, tokens.no_id))
    return F.cross_entropy(logits.float(), targets.long())


def fraud_probability(logits: torch.Tensor, tokens: LabelTokens) -> torch.Tensor:
    """Eq. 14."""
    pair = torch.stack([logits[:, tokens.yes_id], logits[:, tokens.no_id]], dim=-1).float()
    return torch.softmax(pair, dim=-1)[:, 0]


def render_chat_prompt(tokenizer, prompt: str) -> str:
    """One user turn plus the generation prompt. The answer must be the very next token."""
    return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                         add_generation_prompt=True)


def resolve_label_tokens(tokenizer, rendered_prompt: str, positive: str = "Yes", negative: str = "No") -> LabelTokens:
    """Token ids of the two labels, checked in the exact position they are read."""
    prompt_ids = tokenizer.encode(rendered_prompt, add_special_tokens=False)
    resolved = []
    for label in (positive, negative):
        ids = tokenizer.encode(label, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"label {label!r} is {len(ids)} tokens; first-token readout needs one")
        joined = tokenizer.encode(rendered_prompt + label, add_special_tokens=False)
        if joined[:-1] != prompt_ids or joined[-1] != ids[0]:
            raise RuntimeError(f"label {label!r} is not one appended token after the rendered prompt")
        resolved.append(ids[0])
    if resolved[0] == resolved[1]:
        raise RuntimeError("the two labels map to one token")
    return LabelTokens(*resolved)


class LoraClassifier:
    """Gemma-2-9B in bf16 with LoRA on all attention projections."""

    def __init__(self, cfg: dict, device: str):
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg, self.device = cfg, device
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], revision=cfg.get("revision"))
        self.tokenizer.padding_side = "left"   # the answer position is then always the last index
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        attention = {"attn_implementation": cfg["attn_implementation"]} if cfg.get("attn_implementation") else {}
        model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], revision=cfg.get("revision"), dtype=dtype,
                                                     **attention).to(device)
        if cfg.get("gradient_checkpointing", True):
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        lora = cfg["lora"]
        self.model = get_peft_model(model, LoraConfig(r=lora["rank"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
                                                      target_modules=list(lora["target_modules"]), bias="none",
                                                      task_type="CAUSAL_LM"))
        wrapped = {name.rsplit(".", 1)[-1] for name, _ in self.model.named_modules()}
        missing = set(lora["target_modules"]) - wrapped
        if missing:
            raise RuntimeError(f"LoRA target modules not found in the model: {missing}")
        self.labels = resolve_label_tokens(self.tokenizer,
                                           render_chat_prompt(self.tokenizer, "Target: example\nQuestion: Is this fraud?"))

    def last_token_logits(self, prompts: list[str]) -> torch.Tensor:
        rendered = [render_chat_prompt(self.tokenizer, p) for p in prompts]
        encoded = self.tokenizer(rendered, return_tensors="pt", padding=True, add_special_tokens=False)
        if encoded["input_ids"].shape[1] > self.cfg["max_seq_len"]:
            raise ValueError(f"prompt of {encoded['input_ids'].shape[1]} tokens exceeds max_seq_len")
        return self.model(**encoded.to(self.device), logits_to_keep=1).logits[:, -1, :]

    def parameters_to_train(self) -> list[torch.Tensor]:
        return [p for p in self.model.parameters() if p.requires_grad]

    def set_training(self, training: bool) -> None:
        self.model.train(training)

    def state(self) -> dict:
        return {k: v.detach().to("cpu", copy=True) for k, v in self.model.named_parameters() if v.requires_grad}

    def load_state(self, state: dict) -> None:
        parameters = dict(self.model.named_parameters())
        with torch.no_grad():
            for key, value in state.items():
                parameters[key].copy_(value.to(parameters[key].device))

    def save(self, directory: str | Path) -> None:
        self.model.save_pretrained(str(directory))


class MockClassifier(nn.Module):
    """Linear bag-of-words model with the same interface, so tests run the real training path on CPU."""

    labels = LabelTokens(yes_id=2, no_id=3)

    def __init__(self, cfg: dict | None = None, device: str = "cpu", size: int = 512, vocabulary: int = 32):
        super().__init__()
        self.size = size
        self.head = nn.Linear(size, vocabulary)

    def last_token_logits(self, prompts: list[str]) -> torch.Tensor:
        x = torch.zeros(len(prompts), self.size)
        for i, prompt in enumerate(prompts):
            for word in prompt.lower().split():
                x[i, zlib.crc32(word.encode()) % self.size] += 1.0
        return self.head(x / x.sum(dim=1, keepdim=True).clamp(min=1.0))

    def parameters_to_train(self) -> list[torch.Tensor]:
        return list(self.parameters())

    def set_training(self, training: bool) -> None:
        self.train(training)

    def state(self) -> dict:
        return {k: v.detach().clone() for k, v in self.state_dict().items()}

    def load_state(self, state: dict) -> None:
        self.load_state_dict(state)

    def save(self, directory: str | Path) -> None:
        pass


def make_classifier(cfg: dict, device: str):
    return (MockClassifier if cfg.get("backend") == "mock" else LoraClassifier)(cfg, device)
