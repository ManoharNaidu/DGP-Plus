import pytest
import torch

from dgp_plus.classifier import LabelTokens, first_token_loss, fraud_probability, resolve_label_tokens

TOKENS = LabelTokens(yes_id=1, no_id=2)


class Tokenizer:
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary

    def encode(self, text, add_special_tokens=False):
        ids, rest = [], text
        while rest:
            piece = next(p for p in sorted(self.vocabulary, key=len, reverse=True) if rest.startswith(p))
            ids.append(self.vocabulary[piece])
            rest = rest[len(piece):]
        return ids


def test_loss_uses_the_full_vocabulary():
    logits = torch.tensor([[9.0, 0.0, 0.0, 0.0]])                # a non-label token dominates
    assert float(first_token_loss(logits, torch.tensor([1]), TOKENS)) > 5


def test_probability_uses_only_the_two_labels():
    logits = torch.tensor([[100.0, 1.0, 1.0, 0.0]])              # the non-label token is ignored
    assert float(fraud_probability(logits, TOKENS)[0]) == pytest.approx(0.5)


def test_label_tokens_are_checked_at_the_read_position():
    tokenizer = Tokenizer({"\n": 0, "Yes": 1, "No": 2, "x": 3})
    assert resolve_label_tokens(tokenizer, "x\n") == LabelTokens(1, 2)


def test_multi_token_label_stops_the_run():
    with pytest.raises(RuntimeError):
        resolve_label_tokens(Tokenizer({"Y": 1, "es": 4, "No": 2, "\n": 0}), "\n")
