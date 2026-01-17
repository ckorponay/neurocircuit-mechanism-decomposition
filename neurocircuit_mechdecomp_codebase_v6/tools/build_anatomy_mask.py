"""Utility to build a simple directed anatomical mask for CBGTC-like graphs.

This is a *minimal* helper for preliminary analyses. It constructs an adjacency mask
based on token groups (C,S,GPi,GPe,Th,MB,STN,A,H). You can replace this with a
project-specific atlas-derived adjacency.

Mask convention: mask[src, tgt] == True means src -> tgt is permitted.
"""

from __future__ import annotations

import numpy as np


def build_mask(token_sizes: dict[str, int], allow_within: bool = True) -> np.ndarray:
    """Return (R,R) boolean adjacency mask."""
    order = [k for k in ["C", "S", "GPe", "GPi", "STN", "Th", "MB", "A", "H"] if k in token_sizes]
    offsets = {}
    r = 0
    for k in order:
        offsets[k] = (r, r + token_sizes[k])
        r += token_sizes[k]

    mask = np.zeros((r, r), dtype=bool)

    def connect(src: str, tgt: str):
        if src not in offsets or tgt not in offsets:
            return
        s0, s1 = offsets[src]
        t0, t1 = offsets[tgt]
        mask[s0:s1, t0:t1] = True

    # Canonical directed families (coarse CBGTC-inspired)
    connect("C", "S")
    connect("C", "Th")
    connect("Th", "C")
    connect("S", "GPe")
    connect("S", "GPi")
    connect("GPe", "STN")
    connect("STN", "GPi")
    connect("GPi", "Th")
    connect("MB", "S")
    connect("MB", "C")

    # Optional within-family edges (helps stabilize early pilots)
    if allow_within:
        for k in order:
            connect(k, k)

    # Allow limbic inputs to C/S if present
    connect("A", "C")
    connect("A", "S")
    connect("H", "C")
    connect("H", "S")

    return mask


def token_order(token_sizes: dict[str, int]) -> list[str]:
    return [k for k in ["C", "S", "GPe", "GPi", "STN", "Th", "MB", "A", "H"] if k in token_sizes]
