"""Tests for the ActivationCache string representation."""

import pytest
import torch

from transformer_lens.ActivationCache import ActivationCache

HOOKS_PER_LAYER = [
    "hook_resid_pre",
    "ln1.hook_scale",
    "ln1.hook_normalized",
    "attn.hook_q",
    "attn.hook_k",
    "attn.hook_v",
    "attn.hook_attn_scores",
    "attn.hook_pattern",
    "attn.hook_z",
    "hook_attn_out",
    "hook_resid_mid",
    "ln2.hook_scale",
    "ln2.hook_normalized",
    "mlp.hook_pre",
    "mlp.hook_post",
    "hook_mlp_out",
    "hook_resid_post",
]


def make_cache(n_layers: int) -> ActivationCache:
    """Build a cache with the HookedTransformer key layout (2 + 17 * n_layers + 2 keys)."""
    cache_dict = {"hook_embed": torch.zeros(1), "hook_pos_embed": torch.zeros(1)}
    for layer in range(n_layers):
        for hook in HOOKS_PER_LAYER:
            cache_dict[f"blocks.{layer}.{hook}"] = torch.zeros(1)
    cache_dict["ln_final.hook_scale"] = torch.zeros(1)
    cache_dict["ln_final.hook_normalized"] = torch.zeros(1)
    return ActivationCache(cache_dict, model=None)


def make_flat_cache(n_keys: int) -> ActivationCache:
    """Build a cache with ``n_keys`` generic keys, for tests that only care about the key count."""
    return ActivationCache({f"key_{i}": torch.zeros(1) for i in range(n_keys)}, model=None)


def test_small_cache_prints_full_key_list_unchanged():
    # 2-layer layout used by demos/No_Position_Experiment.ipynb; must match the notebook output.
    cache = make_cache(2)
    assert len(cache) == 38
    assert repr(cache) == f"ActivationCache with keys {list(cache.cache_dict.keys())}"


def test_cache_at_threshold_prints_full_key_list():
    cache = make_flat_cache(ActivationCache._REPR_MAX_KEYS)
    assert repr(cache) == f"ActivationCache with keys {list(cache.cache_dict.keys())}"


def test_empty_cache_repr_unchanged():
    assert repr(ActivationCache({}, model=None)) == "ActivationCache with keys []"


def test_head_and_tail_fit_within_threshold():
    # Truncation never runs below the threshold, so head + tail must leave at least one key omitted.
    head_and_tail = ActivationCache._REPR_HEAD_KEYS + ActivationCache._REPR_TAIL_KEYS
    assert head_and_tail < ActivationCache._REPR_MAX_KEYS


# The truncated strings below spell out the 8-head / 3-tail format on purpose: the exact output is
# the contract, so changing _REPR_HEAD_KEYS or _REPR_TAIL_KEYS should break these tests.
def test_cache_above_threshold_is_truncated():
    cache = make_flat_cache(ActivationCache._REPR_MAX_KEYS + 1)
    text = repr(cache)
    assert text == (
        "ActivationCache with 101 keys ['key_0', 'key_1', 'key_2', 'key_3', 'key_4', 'key_5', "
        "'key_6', 'key_7', ..., 'key_98', 'key_99', 'key_100'] "
        "(90 omitted; use cache.keys() for the full list)"
    )
    assert "'key_50'" not in text


def test_gpt2_small_sized_cache_is_truncated():
    cache = make_cache(12)
    text = repr(cache)
    assert text.startswith("ActivationCache with 208 keys ['hook_embed', 'hook_pos_embed', ")
    assert text.endswith(
        "..., 'blocks.11.hook_resid_post', 'ln_final.hook_scale', 'ln_final.hook_normalized'] "
        "(197 omitted; use cache.keys() for the full list)"
    )


def test_truncated_repr_length_is_bounded():
    assert len(repr(make_flat_cache(2000))) < 500


@pytest.mark.parametrize("n_layers", [2, 12])  # 38 keys (full list) and 208 keys (truncated)
def test_repr_does_not_touch_or_mutate_the_cache(n_layers: int):
    cache = make_cache(n_layers)
    cache_dict_before = cache.cache_dict
    keys_before = list(cache.cache_dict.keys())
    tensors_before = list(cache.cache_dict.values())
    flags_before = (cache.has_batch_dim, cache.has_embed, cache.has_pos_embed)
    len_before = len(cache)

    repr(cache)

    assert cache.cache_dict is cache_dict_before
    assert list(cache.cache_dict.keys()) == keys_before
    assert all(
        after is before
        for after, before in zip(cache.cache_dict.values(), tensors_before, strict=True)
    )
    assert (cache.has_batch_dim, cache.has_embed, cache.has_pos_embed) == flags_before
    assert len(cache) == len_before
