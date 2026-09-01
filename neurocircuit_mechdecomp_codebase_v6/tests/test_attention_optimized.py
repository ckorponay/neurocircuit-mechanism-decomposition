import torch

from neurocircuit.models.transformer_head import (
    EdgeConditionedSparseAttention,
    lagged_windows,
)


def test_corrected_lag_window_index_zero_is_current():
    x = torch.tensor([[[0.0, 1.0, 2.0, 3.0]]])
    w = lagged_windows(x, 3, legacy_oldest_first=False)
    assert torch.allclose(w[0, 0, 3], torch.tensor([3.0, 2.0, 1.0]))
    legacy = lagged_windows(x, 3, legacy_oldest_first=True)
    assert torch.allclose(legacy[0, 0, 3], torch.tensor([1.0, 2.0, 3.0]))


def test_edge_conditioning_allows_pathway_specific_lag_peaks():
    attn = EdgeConditionedSparseAttention(
        d_model=1,
        n_lags=2,
        dropout=0.0,
        lag_embedding_mode="index",
        max_lag_seconds=1.0,
        interaction_dim=1,
    )
    with torch.no_grad():
        attn.spatial_q.weight.zero_()
        attn.spatial_k.weight.zero_()
        attn.global_lag_q.weight.zero_()
        attn.edge_q.weight.zero_()
        attn.edge_k.weight.fill_(1.0)
        attn.rel_lag.emb.weight[:, 0] = torch.tensor([-1.0, 1.0])
        attn.interaction_gain.fill_(5.0)

    # Sources 0 and 1 have opposite current embeddings and both project to target 2.
    tokens = torch.zeros(1, 3, 1, 1)
    tokens[0, 0, 0, 0] = 1.0
    tokens[0, 1, 0, 0] = -1.0
    mask = torch.zeros(3, 3, dtype=torch.bool)
    mask[0, 2] = True
    mask[1, 2] = True
    xw = torch.zeros(1, 3, 1, 2)
    lag_idx = torch.tensor([0, 1])
    lag_s = torch.tensor([0.0, 1.0])

    out = attn(tokens, xw, mask, lag_idx, lag_s, return_pi=True)
    peak0 = out["edge_lag_peak_seconds"][0, 0, 2]
    peak1 = out["edge_lag_peak_seconds"][0, 1, 2]
    assert peak0 != peak1
    assert out["pi"].shape == (1, 1, 3, 3, 2)
