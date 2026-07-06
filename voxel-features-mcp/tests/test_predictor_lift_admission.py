from __future__ import annotations

import pytest

from voxel_features import scoring


# Scenarios drawn verbatim from the live Kazakhstan run (2026-06-05). The
# uncalibrated `bic_delta < 0` gate INVERTED the ranking: tiny self-predictive
# blobs admitted (low DOF -> low BIC complexity penalty) while richer, more
# cross-predictive crossbreed children were rejected (high DOF -> high penalty),
# despite genuinely better cross-layer predictor lift. 25 consecutive crossbreed
# failures, KG frozen at 7.
#
# Current policy: admission gates on cross-layer predictor lift (validity + any
# strictly positive lift). bic_delta is telemetry here -- task-level KG admission
# can layer a raw-BIC gate without changing the scorer's lift decision.
_ADMIT_MIN_LIFT = 0.0


@pytest.mark.parametrize(
    "label, validity, lift_mean, bic_delta, expected",
    [
        # tiny kirey blob: marginal positive lift. Under exact >0.0 this clears
        # the scorer's lift gate; novelty/BIC gates live outside this helper.
        ("tiny_blob_marginal_lift", True, 0.0026, -0.075, True),
        # richest-lift crossbreed child (+0.020) but positive bic_delta.
        # OLD rule REJECTED (bic>0); NEW rule admits on lift.
        ("rich_best_lift", True, 0.020, 0.015, True),
        # most-distributed crossbreed child (338 vox), positive bic_delta +0.19.
        # OLD rule REJECTED; NEW rule admits -- this is the layer we most want.
        ("richest_positive_bic", True, 0.011, 0.19, True),
        # genuinely non-predictive (negative lift): rejected either way.
        ("negative_lift", True, -0.0034, 0.11, False),
        # fails self-validity (e.g. blanket): rejected regardless of lift.
        ("invalid_layer", False, 0.5, -1.0, False),
        # exactly at the bar: strictly-greater => reject.
        ("at_bar_exclusive", True, _ADMIT_MIN_LIFT, -0.5, False),
    ],
)
def test_predictor_lift_admission_policy(label, validity, lift_mean, bic_delta, expected):
    decided = scoring.predictor_lift_admission_decision(
        validity_passed=validity,
        lift_mean=lift_mean,
        bic_delta=bic_delta,
        admit_min_lift=_ADMIT_MIN_LIFT,
    )
    assert decided is expected, label


def test_bic_delta_is_telemetry_not_a_veto():
    # Identical lift above the bar, wildly different bic_delta. The calibrated
    # policy must return the SAME verdict -- bic_delta does not gate admission.
    admit_low_bic = scoring.predictor_lift_admission_decision(
        validity_passed=True, lift_mean=0.02, bic_delta=0.0, admit_min_lift=_ADMIT_MIN_LIFT
    )
    admit_high_bic = scoring.predictor_lift_admission_decision(
        validity_passed=True, lift_mean=0.02, bic_delta=5.0, admit_min_lift=_ADMIT_MIN_LIFT
    )
    assert admit_low_bic is True
    assert admit_high_bic is True
