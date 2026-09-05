"""Per-task FPO runner configs and task registry.

All FPO training configs for every supported task live here.
The TASK_CONFIGS dict maps gym task IDs to config classes.
"""

from isaaclab.utils import configclass

from isaaclab_fpo.rl_cfg import (
    FpoRslRlOnPolicyRunnerCfg,
    FpoRslRlPpoActorCriticCfg,
    FpoRslRlPpoAlgorithmCfg,
)

# ---------------------------------------------------------------------------
# Quadruped locomotion (base defaults: n_samples=16, epochs=16, 1500 iters)
# ---------------------------------------------------------------------------


@configclass
class UnitreeGo2FlatFlowPPORunnerCfg(FpoRslRlOnPolicyRunnerCfg):
    """FPO++ on official Unitree-Go2-Velocity.

    Learner stays FPO++ (flow policy, FPO clip / lr / net). Env interface matches
    Unitree PPO: no running obs-norm (scales are already in the MDP), no extra action clip.
    """

    experiment_name = "unitree_go2_flat_flow"
    empirical_normalization = False
    clip_actions = None
    policy = FpoRslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[768, 768, 768],
        activation="elu",
    )
    algorithm = FpoRslRlPpoAlgorithmCfg()


@configclass
class UnitreeGo2FlatFlowPPORunnerCfgReflow(UnitreeGo2FlatFlowPPORunnerCfg):
    """Uniform online reflow on the student (layer B). Theory logs on, no teacher/KD/adaptive."""

    experiment_name = "unitree_go2_flat_flow_reflow"
    algorithm = FpoRslRlPpoAlgorithmCfg(
        reflow_enabled=True,
        reflow_loss_coef=1.0,
        reflow_n_samples_per_obs=4,
        theory_metrics_enabled=True,
    )


@configclass
class UnitreeGo2FlatFlowPPORunnerCfgReflowAdaptiveLambda(UnitreeGo2FlatFlowPPORunnerCfg):
    """Go2 reflow with λ_r adapted from random-x0 1-vs-64 discretization gap.

    Collection stays random x0 (same as baseline / plain reflow).
    """

    experiment_name = "unitree_go2_reflow_adaptive_lambda"
    algorithm = FpoRslRlPpoAlgorithmCfg(
        reflow_enabled=True,
        reflow_loss_coef=1.0,
        reflow_n_samples_per_obs=4,
        reflow_adaptive_lambda_enabled=True,
        reflow_lambda_min=0.1,
        reflow_lambda_max=1.0,
        reflow_lambda_gap_low=0.05,
        reflow_lambda_gap_high=0.4,
        reflow_lambda_ema=0.9,
        reflow_lambda_warmup_iters=50,
        theory_metrics_enabled=True,
    )


@configclass
class UnitreeGo2FlatFlowPPORunnerCfgReflowRandomX0(UnitreeGo2FlatFlowPPORunnerCfg):
    """Reflow + random-x0 few-step consistency (close PostEval random gap)."""

    experiment_name = "unitree_go2_reflow_random_x0"
    policy = FpoRslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[768, 768, 768],
        activation="elu",
        train_flow_x0_mode="mix",
        train_flow_x0_random_prob=0.5,
    )
    algorithm = FpoRslRlPpoAlgorithmCfg(
        reflow_enabled=True,
        reflow_loss_coef=1.0,
        reflow_n_samples_per_obs=4,
        random_x0_consistency_enabled=True,
        random_x0_consistency_coef=0.1,
        random_x0_consistency_steps=[1, 4, 8],
    )


_GO2_BASELINE_TEACHER_500 = (
    "/workspace/plsy/fgo_test/isaaclab_experiments/logs/isaaclab_fpo/"
    "go2_baseline_500/2026-08-27_14-53-04_2026-08-27_14-52-10_baseline/model_499.pt"
)
_UNITREE_GO2_PPO_TEACHER = (
    "/workspace/fgo_test/unitree_rl_lab/logs/rsl_rl/unitree_go2_velocity/"
    "2026-09-02_17-20-40_unitree_go2_ppo/model_2999.pt"
)


@configclass
class UnitreeGo2FlatFlowPPORunnerCfgReflowTeacherKd(UnitreeGo2FlatFlowPPORunnerCfg):
    """FPO++ + uniform reflow + frozen PPO teacher KD. No RA / adaptive."""

    experiment_name = "unitree_go2_reflow_teacher_kd"
    algorithm = FpoRslRlPpoAlgorithmCfg(
        reflow_enabled=True,
        reflow_loss_coef=1.0,
        reflow_n_samples_per_obs=4,
        reflow_teacher_checkpoint=_UNITREE_GO2_PPO_TEACHER,
        teacher_kind="ppo",
        teacher_kd_enabled=True,
        teacher_kd_coef=0.1,
        teacher_kd_steps=[64, 8, 4],
        teacher_aux_zero_x0_prob=0.25,
        theory_metrics_enabled=True,
    )


@configclass
class UnitreeGo2FlatFlowPPORunnerCfgRewardAware(UnitreeGo2FlatFlowPPORunnerCfg):
    """Advantage-weighted reflow (idea1). No teacher / KD / adaptive. Theory logs on."""

    experiment_name = "unitree_go2_reflow_reward_aware"
    algorithm = FpoRslRlPpoAlgorithmCfg(
        reflow_enabled=True,
        reflow_loss_coef=1.0,
        reflow_n_samples_per_obs=4,
        reflow_mode="reward_aware",
        reflow_advantage_threshold=0.0,
        theory_metrics_enabled=True,
    )


@configclass
class UnitreeGo2FlatFlowPPORunnerCfgKdOnly(UnitreeGo2FlatFlowPPORunnerCfg):
    """FPO++ + PPO few-step KD only. No reflow / RA / adaptive. Theory logs on."""

    experiment_name = "unitree_go2_kd_only"
    algorithm = FpoRslRlPpoAlgorithmCfg(
        reflow_enabled=False,
        reflow_teacher_checkpoint=_UNITREE_GO2_PPO_TEACHER,
        teacher_kind="ppo",
        teacher_kd_enabled=True,
        teacher_kd_coef=0.1,
        teacher_kd_steps=[64, 8, 4],
        teacher_aux_zero_x0_prob=0.25,
        theory_metrics_enabled=True,
    )


@configclass
class UnitreeGo2FlatFlowPPORunnerCfgAdaptiveCompute(UnitreeGo2FlatFlowPPORunnerCfg):
    """Idea 2: state-adaptive compute via step predictor."""

    experiment_name = "unitree_go2_reflow_adaptive_compute"
    policy = FpoRslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[768, 768, 768],
        activation="elu",
        adaptive_compute_enabled=True,
    )
    algorithm = FpoRslRlPpoAlgorithmCfg(
        reflow_enabled=True,
        reflow_loss_coef=1.0,
        reflow_n_samples_per_obs=4,
        adaptive_compute_enabled=True,
        adaptive_compute_loss_coef=0.1,
        adaptive_latency_penalty_coef=1.0,
    )


@configclass
class UnitreeGo2FlatFlowPPORunnerCfgFpoOperator(UnitreeGo2FlatFlowPPORunnerCfg):
    """Idea 3: FPO++-compatible reflow as policy-improvement operator."""

    experiment_name = "unitree_go2_reflow_fpo_operator"
    algorithm = FpoRslRlPpoAlgorithmCfg(
        reflow_enabled=True,
        reflow_loss_coef=1.0,
        reflow_n_samples_per_obs=4,
        reflow_mode="fpo_operator",
        reflow_use_ema_endpoint=True,
    )


@configclass
class UnitreeGo2FlatFlowPPORunnerCfgTheory(UnitreeGo2FlatFlowPPORunnerCfg):
    """Idea 4: theory hooks — log path straightness and discretization gaps."""

    experiment_name = "unitree_go2_reflow_theory"
    algorithm = FpoRslRlPpoAlgorithmCfg(
        reflow_enabled=True,
        reflow_loss_coef=1.0,
        reflow_n_samples_per_obs=4,
        theory_metrics_enabled=True,
    )


@configclass
class UnitreeGo2FlatFlowPPORunnerCfgAllIdeas(UnitreeGo2FlatFlowPPORunnerCfg):
    """Stack reward_aware + adaptive_compute + theory (no fpo_operator)."""

    experiment_name = "unitree_go2_reflow_all_ideas"
    policy = FpoRslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[768, 768, 768],
        activation="elu",
        adaptive_compute_enabled=True,
    )
    algorithm = FpoRslRlPpoAlgorithmCfg(
        reflow_enabled=True,
        reflow_loss_coef=1.0,
        reflow_n_samples_per_obs=4,
        reflow_mode="reward_aware",
        reflow_advantage_threshold=0.0,
        theory_metrics_enabled=True,
        adaptive_compute_enabled=True,
        adaptive_compute_loss_coef=0.1,
        adaptive_latency_penalty_coef=1.0,
    )


def _go2_full_policy(*, adaptive: bool = True) -> FpoRslRlPpoActorCriticCfg:
    return FpoRslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[768, 768, 768],
        activation="elu",
        adaptive_compute_enabled=adaptive,
    )


def _go2_full_algorithm(**overrides) -> FpoRslRlPpoAlgorithmCfg:
    """Go2 all_ideas_teacher_kd recipe (PPO teacher). Theory logs always on."""
    kwargs = dict(
        reflow_enabled=True,
        reflow_loss_coef=1.0,
        reflow_n_samples_per_obs=4,
        reflow_mode="reward_aware",
        reflow_advantage_threshold=0.0,
        reflow_teacher_checkpoint=_UNITREE_GO2_PPO_TEACHER,
        teacher_kind="ppo",
        teacher_kd_enabled=True,
        teacher_kd_coef=0.1,
        # Match the already-trained full run (not the paper default {1,4,8}).
        teacher_kd_steps=[64, 8, 4],
        teacher_aux_zero_x0_prob=0.25,
        theory_metrics_enabled=True,
        adaptive_compute_enabled=True,
        adaptive_compute_loss_coef=0.1,
        adaptive_latency_penalty_coef=1.0,
    )
    kwargs.update(overrides)
    return FpoRslRlPpoAlgorithmCfg(**kwargs)


@configclass
class UnitreeGo2FlatFlowPPORunnerCfgAllIdeasTeacherKd(UnitreeGo2FlatFlowPPORunnerCfg):
    """Full method: RA-reflow + PPO KD + aux zero-mix + adaptive. Theory logs on.

    PPO is an action teacher (KD). Reflow x1 stays on the student flow actor.
    """

    experiment_name = "unitree_go2_all_ideas_ppo_teacher"
    policy = _go2_full_policy()
    algorithm = _go2_full_algorithm()


@configclass
class UnitreeGo2FlatFlowPPORunnerCfgAblateNoKd(UnitreeGo2FlatFlowPPORunnerCfgAllIdeasTeacherKd):
    """Full minus few-step PPO distillation (idea3)."""

    experiment_name = "unitree_go2_ablate_no_kd"
    algorithm = _go2_full_algorithm(teacher_kd_enabled=False, reflow_teacher_checkpoint="")


@configclass
class UnitreeGo2FlatFlowPPORunnerCfgAblateNoRa(UnitreeGo2FlatFlowPPORunnerCfgAllIdeasTeacherKd):
    """Full minus advantage-weighted reflow (idea1 → uniform)."""

    experiment_name = "unitree_go2_ablate_no_ra"
    algorithm = _go2_full_algorithm(reflow_mode="uniform")


@configclass
class UnitreeGo2FlatFlowPPORunnerCfgAblateNoAdaptive(UnitreeGo2FlatFlowPPORunnerCfgAllIdeasTeacherKd):
    """Full minus adaptive compute head (idea5)."""

    experiment_name = "unitree_go2_ablate_no_adaptive"
    policy = _go2_full_policy(adaptive=False)
    algorithm = _go2_full_algorithm(adaptive_compute_enabled=False)


@configclass
class UnitreeGo2FlatFlowPPORunnerCfgAblateNoZeroMix(UnitreeGo2FlatFlowPPORunnerCfgAllIdeasTeacherKd):
    """Full minus auxiliary x0=0 mix (idea4). Collection stays random."""

    experiment_name = "unitree_go2_ablate_no_zero_mix"
    algorithm = _go2_full_algorithm(teacher_aux_zero_x0_prob=0.0)


@configclass
class UnitreeGo2FlatFlowPPORunnerCfgAblateNoReflow(UnitreeGo2FlatFlowPPORunnerCfgAllIdeasTeacherKd):
    """Full minus online reflow (layer B). KD + adaptive remain."""

    experiment_name = "unitree_go2_ablate_no_reflow"
    algorithm = _go2_full_algorithm(reflow_enabled=False)


# FPO++ (Yi et al.): flow policy + FPO trust region. reflow_enabled stays False.
GO2_FPO_PLUSPLUS = {
    "baseline": UnitreeGo2FlatFlowPPORunnerCfg,
}
# Author research ideas on top of FPO++ (NOT part of FPO++). Default off.
GO2_REFLOW_IDEAS = {
    "reflow": UnitreeGo2FlatFlowPPORunnerCfgReflow,
    "reflow_adaptive_lambda": UnitreeGo2FlatFlowPPORunnerCfgReflowAdaptiveLambda,
    "reflow_random_x0": UnitreeGo2FlatFlowPPORunnerCfgReflowRandomX0,
    "reflow_teacher_kd": UnitreeGo2FlatFlowPPORunnerCfgReflowTeacherKd,
    "reward_aware": UnitreeGo2FlatFlowPPORunnerCfgRewardAware,
    "kd_only": UnitreeGo2FlatFlowPPORunnerCfgKdOnly,
    # Alias: FPO++ + PPO teacher (trained as kd_only, logged under baseline_plus_teacher).
    "baseline_plus_teacher": UnitreeGo2FlatFlowPPORunnerCfgKdOnly,
    "adaptive_compute": UnitreeGo2FlatFlowPPORunnerCfgAdaptiveCompute,
    "fpo_operator": UnitreeGo2FlatFlowPPORunnerCfgFpoOperator,
    "theory": UnitreeGo2FlatFlowPPORunnerCfgTheory,
    "all_ideas": UnitreeGo2FlatFlowPPORunnerCfgAllIdeas,
    "all_ideas_teacher_kd": UnitreeGo2FlatFlowPPORunnerCfgAllIdeasTeacherKd,
    "ablate_no_kd": UnitreeGo2FlatFlowPPORunnerCfgAblateNoKd,
    "ablate_no_ra": UnitreeGo2FlatFlowPPORunnerCfgAblateNoRa,
    "ablate_no_adaptive": UnitreeGo2FlatFlowPPORunnerCfgAblateNoAdaptive,
    "ablate_no_zero_mix": UnitreeGo2FlatFlowPPORunnerCfgAblateNoZeroMix,
    "ablate_no_reflow": UnitreeGo2FlatFlowPPORunnerCfgAblateNoReflow,
}
GO2_FPO_VARIANTS = {**GO2_FPO_PLUSPLUS, **GO2_REFLOW_IDEAS}

@configclass
class SpotFlatFlowPPORunnerCfg(FpoRslRlOnPolicyRunnerCfg):
    """Spot quadruped: 32 samples, 32 epochs, value_loss_coef=0.5, 1500 iters."""

    experiment_name = "spot_flat_flow"
    policy = FpoRslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[768, 768, 768],
        activation="elu",
    )
    algorithm = FpoRslRlPpoAlgorithmCfg(
        n_samples_per_action=32,
        num_learning_epochs=32,
        value_loss_coef=0.5,
    )


@configclass
class SpotFlatFlowPPORunnerCfgRewardAware(SpotFlatFlowPPORunnerCfg):
    """Spot idea 1: reward-aware rectification on high-advantage samples."""

    experiment_name = "spot_reflow_reward_aware"
    algorithm = FpoRslRlPpoAlgorithmCfg(
        n_samples_per_action=32,
        num_learning_epochs=32,
        value_loss_coef=0.5,
        reflow_enabled=True,
        reflow_loss_coef=1.0,
        reflow_n_samples_per_obs=4,
        reflow_mode="reward_aware",
        reflow_advantage_threshold=0.0,
    )


@configclass
class SpotFlatFlowPPORunnerCfgAdaptiveCompute(SpotFlatFlowPPORunnerCfg):
    """Spot idea 2: state-adaptive compute via step predictor."""

    experiment_name = "spot_reflow_adaptive_compute"
    policy = FpoRslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[768, 768, 768],
        activation="elu",
        adaptive_compute_enabled=True,
    )
    algorithm = FpoRslRlPpoAlgorithmCfg(
        n_samples_per_action=32,
        num_learning_epochs=32,
        value_loss_coef=0.5,
        reflow_enabled=True,
        reflow_loss_coef=1.0,
        reflow_n_samples_per_obs=4,
        adaptive_compute_enabled=True,
        adaptive_compute_loss_coef=0.1,
        adaptive_latency_penalty_coef=1.0,
    )


@configclass
class SpotFlatFlowPPORunnerCfgFpoOperator(SpotFlatFlowPPORunnerCfg):
    """Spot idea 3: FPO++-compatible reflow as policy-improvement operator."""

    experiment_name = "spot_reflow_fpo_operator"
    algorithm = FpoRslRlPpoAlgorithmCfg(
        n_samples_per_action=32,
        num_learning_epochs=32,
        value_loss_coef=0.5,
        reflow_enabled=True,
        reflow_loss_coef=1.0,
        reflow_n_samples_per_obs=4,
        reflow_mode="fpo_operator",
        reflow_use_ema_endpoint=True,
    )


@configclass
class SpotFlatFlowPPORunnerCfgTheory(SpotFlatFlowPPORunnerCfg):
    """Spot idea 4: theory hooks — log path straightness and discretization gaps."""

    experiment_name = "spot_reflow_theory"
    algorithm = FpoRslRlPpoAlgorithmCfg(
        n_samples_per_action=32,
        num_learning_epochs=32,
        value_loss_coef=0.5,
        reflow_enabled=True,
        reflow_loss_coef=1.0,
        reflow_n_samples_per_obs=4,
        theory_metrics_enabled=True,
    )


@configclass
class SpotFlatFlowPPORunnerCfgReflowRandomX0(SpotFlatFlowPPORunnerCfg):
    """Spot: reflow + random-x0 few-step consistency."""

    experiment_name = "spot_reflow_random_x0"
    policy = FpoRslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[768, 768, 768],
        activation="elu",
        train_flow_x0_mode="mix",
        train_flow_x0_random_prob=0.5,
    )
    algorithm = FpoRslRlPpoAlgorithmCfg(
        n_samples_per_action=32,
        num_learning_epochs=32,
        value_loss_coef=0.5,
        reflow_enabled=True,
        reflow_loss_coef=1.0,
        reflow_n_samples_per_obs=4,
        random_x0_consistency_enabled=True,
        random_x0_consistency_coef=0.1,
        random_x0_consistency_steps=[1, 4, 8],
    )


def _unitree_humanoid_full_policy(*, adaptive: bool = True) -> FpoRslRlPpoActorCriticCfg:
    return FpoRslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[768, 768, 768],
        activation="elu",
        adaptive_compute_enabled=adaptive,
    )


def _unitree_ppo_teacher_algorithm(**overrides) -> FpoRslRlPpoAlgorithmCfg:
    """Full method (ours): RA-reflow + PPO teacher KD. Teacher path from --teacher_checkpoint."""
    kwargs = dict(
        n_samples_per_action=32,
        num_learning_epochs=32,
        reflow_enabled=True,
        reflow_loss_coef=1.0,
        reflow_n_samples_per_obs=4,
        reflow_mode="reward_aware",
        reflow_advantage_threshold=0.0,
        reflow_teacher_checkpoint="",
        teacher_kind="ppo",
        teacher_kd_enabled=True,
        teacher_kd_coef=0.1,
        teacher_kd_steps=[64, 8, 4],
        teacher_aux_zero_x0_prob=0.25,
        theory_metrics_enabled=True,
        adaptive_compute_enabled=True,
        adaptive_compute_loss_coef=0.1,
        adaptive_latency_penalty_coef=1.0,
    )
    kwargs.update(overrides)
    return FpoRslRlPpoAlgorithmCfg(**kwargs)


def _spot_ppo_teacher_algorithm(**overrides) -> FpoRslRlPpoAlgorithmCfg:
    kwargs = dict(
        n_samples_per_action=32,
        num_learning_epochs=32,
        value_loss_coef=0.5,
        reflow_enabled=True,
        reflow_loss_coef=1.0,
        reflow_n_samples_per_obs=4,
        reflow_mode="reward_aware",
        reflow_advantage_threshold=0.0,
        reflow_teacher_checkpoint="",
        teacher_kind="ppo",
        teacher_kd_enabled=True,
        teacher_kd_coef=0.1,
        teacher_kd_steps=[64, 8, 4],
        teacher_aux_zero_x0_prob=0.25,
        theory_metrics_enabled=True,
        adaptive_compute_enabled=True,
        adaptive_compute_loss_coef=0.1,
        adaptive_latency_penalty_coef=1.0,
    )
    kwargs.update(overrides)
    return FpoRslRlPpoAlgorithmCfg(**kwargs)


@configclass
class SpotFlatFlowPPORunnerCfgKdOnly(SpotFlatFlowPPORunnerCfg):
    """FPO++ + PPO few-step KD only (no reflow / RA / adaptive). Train from scratch."""

    experiment_name = "spot_kd_only"
    algorithm = FpoRslRlPpoAlgorithmCfg(
        n_samples_per_action=32,
        num_learning_epochs=32,
        value_loss_coef=0.5,
        reflow_enabled=False,
        reflow_teacher_checkpoint="",
        teacher_kind="ppo",
        teacher_kd_enabled=True,
        teacher_kd_coef=0.1,
        teacher_kd_steps=[64, 8, 4],
        teacher_aux_zero_x0_prob=0.25,
        theory_metrics_enabled=True,
    )


@configclass
class SpotFlatFlowPPORunnerCfgAllIdeasTeacherKd(SpotFlatFlowPPORunnerCfg):
    """Spot full method: PPO teacher + RA-reflow + KD + adaptive + theory."""

    experiment_name = "spot_all_ideas_ppo_teacher"
    policy = FpoRslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[768, 768, 768],
        activation="elu",
        adaptive_compute_enabled=True,
    )
    algorithm = _spot_ppo_teacher_algorithm()


SPOT_FPO_VARIANTS = {
    "baseline": SpotFlatFlowPPORunnerCfg,
    "kd_only": SpotFlatFlowPPORunnerCfgKdOnly,
    "baseline_plus_teacher": SpotFlatFlowPPORunnerCfgKdOnly,
    "all_ideas_teacher_kd": SpotFlatFlowPPORunnerCfgAllIdeasTeacherKd,
    "reflow_random_x0": SpotFlatFlowPPORunnerCfgReflowRandomX0,
    "reward_aware": SpotFlatFlowPPORunnerCfgRewardAware,
    "adaptive_compute": SpotFlatFlowPPORunnerCfgAdaptiveCompute,
    "fpo_operator": SpotFlatFlowPPORunnerCfgFpoOperator,
    "theory": SpotFlatFlowPPORunnerCfgTheory,
}


@configclass
class H1FlatFlowPPORunnerCfg(FpoRslRlOnPolicyRunnerCfg):
    """H1 humanoid: 32 samples, 32 epochs, 2000 iters."""

    max_iterations = 2000
    experiment_name = "h1_flat_flow"
    policy = FpoRslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[768, 768, 768],
        activation="elu",
    )
    algorithm = FpoRslRlPpoAlgorithmCfg(
        n_samples_per_action=32,
        num_learning_epochs=32,
    )


@configclass
class G1FlatFlowPPORunnerCfg(FpoRslRlOnPolicyRunnerCfg):
    """G1 humanoid: 32 samples, 32 epochs, 2000 iters."""

    max_iterations = 2000
    experiment_name = "g1_flat_flow"
    policy = FpoRslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[768, 768, 768],
        activation="elu",
    )
    algorithm = FpoRslRlPpoAlgorithmCfg(
        n_samples_per_action=32,
        num_learning_epochs=32,
    )


@configclass
class G1FlatFlowPPORunnerCfgReflow(G1FlatFlowPPORunnerCfg):
    """G1 with Rectified Flow reflow auxiliary loss enabled."""

    experiment_name = "g1_flat_flow_reflow"
    algorithm = FpoRslRlPpoAlgorithmCfg(
        n_samples_per_action=32,
        num_learning_epochs=32,
        reflow_enabled=True,
        reflow_loss_coef=1.0,
        reflow_n_samples_per_obs=4,
    )


_G1_BASELINE_TEACHER_2000 = (
    "/workspace/plsy/fgo_test/isaaclab_experiments/logs/isaaclab_fpo/"
    "g1_flat_flow/2026-08-27_00-04-43_2026-08-27_00-02-39_g1_baseline/model_1999.pt"
)


@configclass
class G1FlatFlowPPORunnerCfgReflowTeacherKd(G1FlatFlowPPORunnerCfg):
    """G1 reflow student with frozen full baseline teacher + few-step KD."""

    experiment_name = "g1_reflow_teacher_kd"
    algorithm = FpoRslRlPpoAlgorithmCfg(
        n_samples_per_action=32,
        num_learning_epochs=32,
        reflow_enabled=True,
        reflow_loss_coef=1.0,
        reflow_n_samples_per_obs=4,
        reflow_teacher_checkpoint=_G1_BASELINE_TEACHER_2000,
        teacher_kd_enabled=True,
        teacher_kd_coef=0.1,
        teacher_kd_steps=[1, 4, 8],
        teacher_aux_zero_x0_prob=0.25,
    )


@configclass
class G1FlatFlowPPORunnerCfgReflowRandomX0(G1FlatFlowPPORunnerCfg):
    """G1: reflow + random-x0 few-step consistency."""

    experiment_name = "g1_reflow_random_x0"
    policy = FpoRslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[768, 768, 768],
        activation="elu",
        train_flow_x0_mode="mix",
        train_flow_x0_random_prob=0.5,
    )
    algorithm = FpoRslRlPpoAlgorithmCfg(
        n_samples_per_action=32,
        num_learning_epochs=32,
        reflow_enabled=True,
        reflow_loss_coef=1.0,
        reflow_n_samples_per_obs=4,
        random_x0_consistency_enabled=True,
        random_x0_consistency_coef=0.1,
        random_x0_consistency_steps=[1, 4, 8],
    )


@configclass
class G1FlatFlowPPORunnerCfgRewardAware(G1FlatFlowPPORunnerCfg):
    """G1 idea 1: reward-aware rectification on high-advantage samples."""

    experiment_name = "g1_reflow_reward_aware"
    algorithm = FpoRslRlPpoAlgorithmCfg(
        n_samples_per_action=32,
        num_learning_epochs=32,
        reflow_enabled=True,
        reflow_loss_coef=1.0,
        reflow_n_samples_per_obs=4,
        reflow_mode="reward_aware",
        reflow_advantage_threshold=0.0,
    )


@configclass
class G1FlatFlowPPORunnerCfgAdaptiveCompute(G1FlatFlowPPORunnerCfg):
    """G1 idea 2: state-adaptive compute via step predictor."""

    experiment_name = "g1_reflow_adaptive_compute"
    policy = FpoRslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[768, 768, 768],
        activation="elu",
        adaptive_compute_enabled=True,
    )
    algorithm = FpoRslRlPpoAlgorithmCfg(
        n_samples_per_action=32,
        num_learning_epochs=32,
        reflow_enabled=True,
        reflow_loss_coef=1.0,
        reflow_n_samples_per_obs=4,
        adaptive_compute_enabled=True,
        adaptive_compute_loss_coef=0.1,
        adaptive_latency_penalty_coef=1.0,
    )


@configclass
class G1FlatFlowPPORunnerCfgFpoOperator(G1FlatFlowPPORunnerCfg):
    """G1 idea 3: FPO++-compatible reflow as policy-improvement operator."""

    experiment_name = "g1_reflow_fpo_operator"
    algorithm = FpoRslRlPpoAlgorithmCfg(
        n_samples_per_action=32,
        num_learning_epochs=32,
        reflow_enabled=True,
        reflow_loss_coef=1.0,
        reflow_n_samples_per_obs=4,
        reflow_mode="fpo_operator",
        reflow_use_ema_endpoint=True,
    )


@configclass
class G1FlatFlowPPORunnerCfgTheory(G1FlatFlowPPORunnerCfg):
    """G1 idea 4: theory hooks — log path straightness and discretization gaps."""

    experiment_name = "g1_reflow_theory"
    algorithm = FpoRslRlPpoAlgorithmCfg(
        n_samples_per_action=32,
        num_learning_epochs=32,
        reflow_enabled=True,
        reflow_loss_coef=1.0,
        reflow_n_samples_per_obs=4,
        theory_metrics_enabled=True,
    )


@configclass
class G1FlatFlowPPORunnerCfgAllIdeas(G1FlatFlowPPORunnerCfg):
    """G1 stack: reward_aware + adaptive_compute + theory (no fpo_operator)."""

    experiment_name = "g1_reflow_all_ideas"
    policy = FpoRslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[768, 768, 768],
        activation="elu",
        adaptive_compute_enabled=True,
    )
    algorithm = FpoRslRlPpoAlgorithmCfg(
        n_samples_per_action=32,
        num_learning_epochs=32,
        reflow_enabled=True,
        reflow_loss_coef=1.0,
        reflow_n_samples_per_obs=4,
        reflow_mode="reward_aware",
        reflow_advantage_threshold=0.0,
        theory_metrics_enabled=True,
        adaptive_compute_enabled=True,
        adaptive_compute_loss_coef=0.1,
        adaptive_latency_penalty_coef=1.0,
    )


@configclass
class G1FlatFlowPPORunnerCfgAllIdeasFpo(G1FlatFlowPPORunnerCfg):
    """G1 stack: fpo_operator + adaptive_compute + theory (fixed EMA endpoint)."""

    experiment_name = "g1_reflow_all_ideas_fpo"
    policy = FpoRslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[768, 768, 768],
        activation="elu",
        adaptive_compute_enabled=True,
    )
    algorithm = FpoRslRlPpoAlgorithmCfg(
        n_samples_per_action=32,
        num_learning_epochs=32,
        reflow_enabled=True,
        reflow_loss_coef=1.0,
        reflow_n_samples_per_obs=4,
        reflow_mode="fpo_operator",
        reflow_use_ema_endpoint=True,
        theory_metrics_enabled=True,
        adaptive_compute_enabled=True,
        adaptive_compute_loss_coef=0.1,
        adaptive_latency_penalty_coef=1.0,
    )


@configclass
class G1FlatFlowPPORunnerCfgAllIdeasTeacherKd(G1FlatFlowPPORunnerCfg):
    """Full G1 stack: frozen baseline teacher + KD + fpo_operator weights + adaptive + theory.

    EMA endpoints are off: the frozen teacher owns x1. Collection stays random.
    """

    experiment_name = "g1_all_ideas_teacher_kd"
    policy = FpoRslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[768, 768, 768],
        activation="elu",
        adaptive_compute_enabled=True,
    )
    algorithm = FpoRslRlPpoAlgorithmCfg(
        n_samples_per_action=32,
        num_learning_epochs=32,
        reflow_enabled=True,
        reflow_loss_coef=1.0,
        reflow_n_samples_per_obs=4,
        reflow_mode="fpo_operator",
        reflow_use_ema_endpoint=False,
        reflow_teacher_checkpoint=_G1_BASELINE_TEACHER_2000,
        teacher_kd_enabled=True,
        teacher_kd_coef=0.1,
        teacher_kd_steps=[1, 4, 8],
        teacher_aux_zero_x0_prob=0.25,
        theory_metrics_enabled=True,
        adaptive_compute_enabled=True,
        adaptive_compute_loss_coef=0.1,
        adaptive_latency_penalty_coef=1.0,
    )


G1_FPO_VARIANTS = {
    "baseline": G1FlatFlowPPORunnerCfg,
    "reflow": G1FlatFlowPPORunnerCfgReflow,
    "reflow_teacher_kd": G1FlatFlowPPORunnerCfgReflowTeacherKd,
    "reflow_random_x0": G1FlatFlowPPORunnerCfgReflowRandomX0,
    "reward_aware": G1FlatFlowPPORunnerCfgRewardAware,
    "adaptive_compute": G1FlatFlowPPORunnerCfgAdaptiveCompute,
    "fpo_operator": G1FlatFlowPPORunnerCfgFpoOperator,
    "theory": G1FlatFlowPPORunnerCfgTheory,
    "all_ideas": G1FlatFlowPPORunnerCfgAllIdeas,
    "all_ideas_fpo": G1FlatFlowPPORunnerCfgAllIdeasFpo,
    "all_ideas_teacher_kd": G1FlatFlowPPORunnerCfgAllIdeasTeacherKd,
}


# ---------------------------------------------------------------------------
# Motion tracking (G1 humanoid, whole-body)
# ---------------------------------------------------------------------------


@configclass
class G1FlatMotionTrackingFlowPPORunnerCfg(FpoRslRlOnPolicyRunnerCfg):
    """G1 whole-body motion tracking: 20k iters, no action clipping, ASPO."""

    num_steps_per_env = 48
    max_iterations = 20000
    save_interval = 500
    clip_actions = None  # No action clipping for tracking (PD targets can exceed joint limits)
    experiment_name = "g1_flat_motion_tracking"
    policy = FpoRslRlPpoActorCriticCfg(
        actor_hidden_dims=[1024, 512, 256],
        critic_hidden_dims=[1024, 512, 256],
        activation="elu",
        cfm_loss_reduction="mean",
        action_perturb_std=0.1,
    )
    algorithm = FpoRslRlPpoAlgorithmCfg(
        clip_param=0.01,
        num_mini_batches=6,
        schedule="adaptive",
        trust_region_mode="aspo",
        cfm_loss_clamp=3.0,
        cfm_diff_clamp_max=3.0,
        advantage_clamp=(5.0, 5.0),
    )


# ---------------------------------------------------------------------------
# Cartpole (direct env, useful for quick debugging)
# ---------------------------------------------------------------------------


@configclass
class CartpoleFlowPPORunnerCfg(FpoRslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16
    max_iterations = 150
    save_interval = 50
    experiment_name = "cartpole_direct"
    empirical_normalization = False
    policy = FpoRslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[32, 32],
        critic_hidden_dims=[32, 32],
        activation="elu",
        actor_scale=1.0,
        actor_mlp_output_scale=1.0,
    )
    algorithm = FpoRslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.03,
        knn_entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        n_samples_per_action=256,
    )


# ---------------------------------------------------------------------------
# Task registry: maps gym task ID -> FPO config class
# ---------------------------------------------------------------------------

@configclass
class UnitreeG1FlatFlowPPORunnerCfg(FpoRslRlOnPolicyRunnerCfg):
    """FPO++ on official Unitree-G1-29dof-Velocity (same MDP interface as Unitree PPO)."""

    experiment_name = "unitree_g1_flat_flow"
    empirical_normalization = False
    clip_actions = None
    max_iterations = 2000
    policy = FpoRslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[768, 768, 768],
        activation="elu",
    )
    algorithm = FpoRslRlPpoAlgorithmCfg(
        n_samples_per_action=32,
        num_learning_epochs=32,
    )


@configclass
class UnitreeG1FlatFlowPPORunnerCfgKdOnly(UnitreeG1FlatFlowPPORunnerCfg):
    """FPO++ + PPO few-step KD only. Train from scratch (pass --teacher_checkpoint)."""

    experiment_name = "unitree_g1_kd_only"
    algorithm = FpoRslRlPpoAlgorithmCfg(
        n_samples_per_action=32,
        num_learning_epochs=32,
        reflow_enabled=False,
        reflow_teacher_checkpoint="",
        teacher_kind="ppo",
        teacher_kd_enabled=True,
        teacher_kd_coef=0.1,
        teacher_kd_steps=[64, 8, 4],
        teacher_aux_zero_x0_prob=0.25,
        theory_metrics_enabled=True,
    )


@configclass
class UnitreeG1FlatFlowPPORunnerCfgAllIdeasTeacherKd(UnitreeG1FlatFlowPPORunnerCfg):
    """Official G1 full method: PPO teacher + RA-reflow + KD + adaptive + theory."""

    experiment_name = "unitree_g1_all_ideas_ppo_teacher"
    policy = _unitree_humanoid_full_policy()
    algorithm = _unitree_ppo_teacher_algorithm()


@configclass
class UnitreeH1FlatFlowPPORunnerCfg(FpoRslRlOnPolicyRunnerCfg):
    """FPO++ on official Unitree-H1-Velocity."""

    experiment_name = "unitree_h1_flat_flow"
    empirical_normalization = False
    clip_actions = None
    max_iterations = 2000
    policy = FpoRslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[768, 768, 768],
        activation="elu",
    )
    algorithm = FpoRslRlPpoAlgorithmCfg(
        n_samples_per_action=32,
        num_learning_epochs=32,
    )


@configclass
class UnitreeH1FlatFlowPPORunnerCfgKdOnly(UnitreeH1FlatFlowPPORunnerCfg):
    """FPO++ + PPO few-step KD only. Train from scratch (pass --teacher_checkpoint)."""

    experiment_name = "unitree_h1_kd_only"
    algorithm = FpoRslRlPpoAlgorithmCfg(
        n_samples_per_action=32,
        num_learning_epochs=32,
        reflow_enabled=False,
        reflow_teacher_checkpoint="",
        teacher_kind="ppo",
        teacher_kd_enabled=True,
        teacher_kd_coef=0.1,
        teacher_kd_steps=[64, 8, 4],
        teacher_aux_zero_x0_prob=0.25,
        theory_metrics_enabled=True,
    )


@configclass
class UnitreeH1FlatFlowPPORunnerCfgAllIdeasTeacherKd(UnitreeH1FlatFlowPPORunnerCfg):
    """Official H1 full method: PPO teacher + RA-reflow + KD + adaptive + theory."""

    experiment_name = "unitree_h1_all_ideas_ppo_teacher"
    policy = _unitree_humanoid_full_policy()
    algorithm = _unitree_ppo_teacher_algorithm()


UNITREE_G1_FPO_VARIANTS = {
    "baseline": UnitreeG1FlatFlowPPORunnerCfg,
    "kd_only": UnitreeG1FlatFlowPPORunnerCfgKdOnly,
    "baseline_plus_teacher": UnitreeG1FlatFlowPPORunnerCfgKdOnly,
    "all_ideas_teacher_kd": UnitreeG1FlatFlowPPORunnerCfgAllIdeasTeacherKd,
}

UNITREE_H1_FPO_VARIANTS = {
    "baseline": UnitreeH1FlatFlowPPORunnerCfg,
    "kd_only": UnitreeH1FlatFlowPPORunnerCfgKdOnly,
    "baseline_plus_teacher": UnitreeH1FlatFlowPPORunnerCfgKdOnly,
    "all_ideas_teacher_kd": UnitreeH1FlatFlowPPORunnerCfgAllIdeasTeacherKd,
}


TASK_CONFIGS = {
    # Official unitree_rl_lab Go2 env → FPO++ only (baseline). Reflow ideas via --fpo_variant.
    "Unitree-Go2-Velocity": UnitreeGo2FlatFlowPPORunnerCfg,
    "Unitree-G1-29dof-Velocity": UnitreeG1FlatFlowPPORunnerCfg,
    "Unitree-H1-Velocity": UnitreeH1FlatFlowPPORunnerCfg,
    # Quadrupeds (original Isaac Lab locomotion tasks; still registered for convenience)
    "Isaac-Velocity-Flat-Unitree-Go2-v0": UnitreeGo2FlatFlowPPORunnerCfg,
    "Isaac-Velocity-Flat-Unitree-Go2-Play-v0": UnitreeGo2FlatFlowPPORunnerCfg,
    "Isaac-Velocity-Flat-Spot-v0": SpotFlatFlowPPORunnerCfg,
    "Isaac-Velocity-Flat-Spot-Play-v0": SpotFlatFlowPPORunnerCfg,
    # Humanoids
    "Isaac-Velocity-Flat-H1-v0": H1FlatFlowPPORunnerCfg,
    "Isaac-Velocity-Flat-H1-Play-v0": H1FlatFlowPPORunnerCfg,
    "Isaac-Velocity-Flat-G1-v0": G1FlatFlowPPORunnerCfg,
    "Isaac-Velocity-Flat-G1-Play-v0": G1FlatFlowPPORunnerCfg,
    # Motion tracking
    "Tracking-Flat-G1-v0": G1FlatMotionTrackingFlowPPORunnerCfg,
    # Direct envs
    "Isaac-Cartpole-Direct-v0": CartpoleFlowPPORunnerCfg,
}
