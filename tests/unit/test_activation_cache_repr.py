"""Unit tests for ``ActivationCache.__repr__`` and the typed ``__getitem__`` key handling.

These live in the ``unit`` tier because they exercise isolated behaviour of the cache object with
synthetic tensors — no model is loaded and the HF Hub is never touched. ``ActivationCache`` only
needs ``model.cfg.n_layers`` for negative-layer lookups, so a ``SimpleNamespace`` stands in for the
model.

The regression guard for the slow-repr bug is ``test_repr_is_bounded_for_large_cache``: the old
implementation listed every key, so its output grew linearly with the cache and displaying a
many-thousand-key cache in a notebook effectively hung the front-end.
"""

from types import SimpleNamespace

import pytest
import torch

from transformer_lens.ActivationCache import ActivationCache

_HOOKS_PER_LAYER = (
    "hook_resid_pre",
    "attn.hook_q",
    "attn.hook_k",
    "attn.hook_v",
    "attn.hook_z",
    "attn.hook_pattern",
    "hook_attn_out",
    "hook_resid_mid",
    "mlp.hook_pre",
    "mlp.hook_post",
    "hook_mlp_out",
    "hook_resid_post",
)


def _make_cache(n_layers: int, batch: int = 1, pos: int = 3, d_model: int = 4) -> ActivationCache:
    """Build a cache whose key layout mirrors ``run_with_cache`` but with tiny synthetic tensors."""
    cache_dict = {
        "hook_embed": torch.zeros(batch, pos, d_model),
        "hook_pos_embed": torch.zeros(batch, pos, d_model),
    }
    for layer in range(n_layers):
        for hook in _HOOKS_PER_LAYER:
            cache_dict[f"blocks.{layer}.{hook}"] = torch.zeros(batch, pos, d_model)
    cache_dict["ln_final.hook_normalized"] = torch.zeros(batch, pos, d_model)
    model = SimpleNamespace(cfg=SimpleNamespace(n_layers=n_layers))
    return ActivationCache(cache_dict, model, has_batch_dim=True)


def test_repr_reports_count_batch_dim_and_device():
    cache = _make_cache(n_layers=2)
    rep = repr(cache)
    assert rep.startswith("ActivationCache(")
    assert f"n_activations={len(cache)}" in rep
    assert "has_batch_dim=True" in rep
    assert "device='cpu'" in rep


def test_repr_previews_first_keys_and_elides_the_rest():
    cache = _make_cache(n_layers=2)
    rep = repr(cache)
    # The first keys are shown so the repr is still recognisable...
    assert "'hook_embed'" in rep
    assert "'hook_pos_embed'" in rep
    # ...but the tail is elided with a count rather than listed.
    assert "'ln_final.hook_normalized'" not in rep
    assert "more)" in rep
    assert rep.count("'") < 2 * len(cache)


def test_repr_small_cache_lists_all_keys_without_ellipsis():
    cache_dict = {"hook_embed": torch.zeros(1, 2, 4), "hook_pos_embed": torch.zeros(1, 2, 4)}
    cache = ActivationCache(cache_dict, SimpleNamespace(cfg=SimpleNamespace(n_layers=0)))
    rep = repr(cache)
    assert "'hook_embed', 'hook_pos_embed'" in rep
    assert "more)" not in rep


def test_repr_reflects_removed_batch_dim():
    cache = _make_cache(n_layers=1)
    cache.remove_batch_dim()
    assert "has_batch_dim=False" in repr(cache)


def test_repr_empty_cache():
    cache = ActivationCache({}, SimpleNamespace(cfg=SimpleNamespace(n_layers=0)))
    rep = repr(cache)
    assert "n_activations=0" in rep
    assert "keys=[]" in rep
    assert "device=" not in rep


def test_repr_is_bounded_for_large_cache():
    """Regression guard for the slow repr: output size must not scale with the number of keys.

    The old implementation listed every key, so a ~4.8k-key cache produced a ~130 KB repr. The
    check is on output size rather than wall-clock time, which is flaky on CI runners.
    """
    small = _make_cache(n_layers=2)
    large = _make_cache(n_layers=400)  # ~4.8k keys, comparable to a big model in compat mode
    assert len(large) > 4000

    small_rep = repr(small)
    large_rep = repr(large)

    # Only the two counts (n_activations and the "N more" tail) may be longer for the big cache.
    assert len(large_rep) < len(small_rep) + 32
    assert len(large_rep) < 512
    # The elided keys are not present anywhere in the output.
    assert "blocks.399" not in large_rep
    assert "ln_final.hook_normalized" not in large_rep
    assert f"({len(large) - 5} more)" in large_rep


def test_getitem_accepts_string_and_tuple_keys():
    cache = _make_cache(n_layers=3)
    target = cache.cache_dict["blocks.1.attn.hook_z"]
    assert cache["blocks.1.attn.hook_z"] is target
    assert cache["z1"] is target
    assert cache[("z", 1)] is target
    assert cache[("z", 1, "attn")] is target
    assert cache[("embed",)] is cache.cache_dict["hook_embed"]


def test_getitem_negative_layer_index_counts_from_end():
    cache = _make_cache(n_layers=3)
    assert cache[("resid_post", -1)] is cache.cache_dict["blocks.2.hook_resid_post"]
    assert cache[("z", -3, "attn")] is cache.cache_dict["blocks.0.attn.hook_z"]


def test_getitem_missing_key_raises_key_error():
    cache = _make_cache(n_layers=1)
    with pytest.raises(KeyError):
        cache["blocks.7.hook_resid_post"]
