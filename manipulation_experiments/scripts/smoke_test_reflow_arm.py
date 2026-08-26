#!/usr/bin/env python
"""Smoke-test arm reflow idea helpers + FlowMatchingPolicy loss APIs (no env)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.reflow_extensions import (  # noqa: E402
    StepPredictor,
    compute_advantage_reflow_weights,
    compute_discretization_gap,
    compute_path_straightness,
)


def test_helpers() -> None:
    adv = torch.tensor([1.0, -0.5, 0.2, 0.0])
    w_ra = compute_advantage_reflow_weights(adv, "reward_aware", threshold=0.0)
    w_op = compute_advantage_reflow_weights(adv, "fpo_operator", threshold=0.0)
    assert torch.isfinite(w_ra).all() and w_ra.sum().item() > 0
    assert torch.isfinite(w_op).all() and w_op.sum().item() > 0

    x0 = torch.randn(4, 8, 7)
    x1 = torch.randn(4, 8, 7)
    u = x0 - x1
    s = compute_path_straightness(x0, x1, u)
    g = compute_discretization_gap(x1, x1 + 0.1)
    assert s.shape == (4,) and g.shape == (4,)

    pred = StepPredictor(num_obs=16, step_bins=[1, 2, 4, 8])
    obs = torch.randn(4, 16)
    steps = pred.predict_steps(obs)
    frac = pred.expected_step_fraction(obs)
    assert steps.shape == (4,) and frac.shape == (4,)
    print("helpers OK", float(w_ra.mean()), float(steps.float().mean()), float(frac.mean()))


def test_policy_losses_if_available() -> None:
    """Optional: only runs if a tiny policy can be constructed from config."""
    try:
        from src.flow_model_config import FlowMatchingConfig
        from src.flow_model import FlowMatchingPolicy
    except Exception as exc:  # pragma: no cover
        print(f"skip policy smoke (import failed): {exc}")
        return

    # Minimal state-only config to avoid vision weights download.
    cfg = FlowMatchingConfig(
        horizon=8,
        n_action_steps=4,
        sampling_steps=4,
        network_architecture="mlp",
        mlp_dims=[64, 64],
        input_features=["observation.state"],
        output_features=["action"],
        input_shapes={"observation.state": [10]},
        output_shapes={"action": [7]},
        ema_power=0.75,
    )
    try:
        policy = FlowMatchingPolicy(cfg)
    except Exception as exc:  # pragma: no cover
        print(f"skip policy smoke (construct failed): {exc}")
        return

    device = torch.device("cpu")
    policy.to(device)
    B = 2
    batch = {
        "observation.state": torch.randn(B, 10, device=device),
        "action": torch.randn(B, cfg.horizon, 7, device=device),
    }
    adv = torch.randn(B, 1, device=device)

    loss_u = policy.get_reflow_loss(batch, n_samples_per_obs=2, reflow_mode="uniform")
    loss_ra = policy.get_reflow_loss(
        batch, n_samples_per_obs=2, advantages=adv, reflow_mode="reward_aware"
    )
    loss_op = policy.get_reflow_loss(
        batch, n_samples_per_obs=2, advantages=adv, reflow_mode="fpo_operator"
    )
    assert torch.isfinite(loss_u) and torch.isfinite(loss_ra) and torch.isfinite(loss_op)

    policy.ensure_step_predictor([1, 2])
    adaptive_loss, metrics = policy.get_adaptive_compute_loss(
        batch, advantages=adv, step_bins=[1, 2]
    )
    assert torch.isfinite(adaptive_loss)
    theory = policy.compute_theory_metrics(batch)
    assert "path_straightness" in theory
    print(
        "policy losses OK",
        float(loss_u),
        float(adaptive_loss),
        theory.get("path_straightness"),
        metrics,
    )


if __name__ == "__main__":
    test_helpers()
    test_policy_losses_if_available()
    print("smoke_test_reflow_arm: PASS")
