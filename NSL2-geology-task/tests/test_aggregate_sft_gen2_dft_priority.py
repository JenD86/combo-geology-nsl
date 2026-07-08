"""Tests for the gen-2 DFT-first corpus composition (pure logic; no nix/transform deps).

Covers the only non-trivial logic in the aggregator: DFT-priority + target cap + shortfall.
Run: python3 -m pytest tests/test_aggregate_sft_gen2_dft_priority.py -q
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from aggregate_sft_gen2_dft_priority import compose_dft_first  # noqa: E402


def _rows(tag, n):
    return [{"prompt": f"{tag}{i}", "raw_response": "x", "source_run_id": tag} for i in range(n)]


def test_dft_under_target_fills_with_rmx():
    dft = _rows("dft", 3)
    rmx = _rows("rmx", 5)
    pooled, st = compose_dft_first(dft, rmx, target=6)
    assert st["dft_used"] == 3 and st["rmx_used"] == 3 and st["total"] == 6
    assert st["dft_capped"] is False and st["shortfall"] == 0
    # every DFT row is present, RMX only fills the remainder, DFT comes first
    assert [r["source_run_id"] for r in pooled] == ["dft"] * 3 + ["rmx"] * 3


def test_dft_over_target_is_capped_no_rmx():
    dft = _rows("dft", 8)
    rmx = _rows("rmx", 5)
    pooled, st = compose_dft_first(dft, rmx, target=6)
    assert st["dft_used"] == 6 and st["rmx_used"] == 0 and st["total"] == 6
    assert st["dft_capped"] is True
    assert all(r["source_run_id"] == "dft" for r in pooled)


def test_dft_exactly_target_uses_no_rmx():
    dft = _rows("dft", 6)
    rmx = _rows("rmx", 5)
    pooled, st = compose_dft_first(dft, rmx, target=6)
    assert st["dft_used"] == 6 and st["rmx_used"] == 0 and st["dft_capped"] is True
    assert len(pooled) == 6


def test_shortfall_reported_when_not_enough_rows():
    dft = _rows("dft", 3)
    rmx = _rows("rmx", 1)
    pooled, st = compose_dft_first(dft, rmx, target=10)
    assert st["total"] == 4 and st["shortfall"] == 6
    assert len(pooled) == 4


def test_inputs_not_mutated():
    dft = _rows("dft", 2)
    rmx = _rows("rmx", 2)
    pooled, _ = compose_dft_first(dft, rmx, target=3)
    assert len(dft) == 2 and len(rmx) == 2  # originals intact
    pooled.append({"x": 1})
    assert len(dft) == 2 and len(rmx) == 2
