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
    """Go2 quadruped: uses base defaults (n_samples=16, epochs=16, 1500 iters)."""

    experiment_name = "unitree_go2_flat_flow"
    policy = FpoRslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[768, 768, 768],
        activation="elu",
    )
    algorithm = FpoRslRlPpoAlgorithmCfg()


@configclass
class UnitreeGo2FlatFlowPPORunnerCfgReflow(UnitreeGo2FlatFlowPPORunnerCfg):
    """Go2 with Rectified Flow reflow auxiliary loss enabled."""

    experiment_name = "unitree_go2_flat_flow_reflow"
    algorithm = FpoRslRlPpoAlgorithmCfg(
        reflow_enabled=True,
        reflow_loss_coef=1.0,
        reflow_n_samples_per_obs=4,
    )


@configclass
class UnitreeGo2FlatFlowPPORunnerCfgRewardAware(UnitreeGo2FlatFlowPPORunnerCfg):
    """Idea 1: reward-aware rectification on high-advantage samples."""

    experiment_name = "unitree_go2_reflow_reward_aware"
    algorithm = FpoRslRlPpoAlgorithmCfg(
        reflow_enabled=True,
        reflow_loss_coef=1.0,
        reflow_n_samples_per_obs=4,
        reflow_mode="reward_aware",
        reflow_advantage_threshold=0.0,
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


GO2_FPO_VARIANTS = {
    "baseline": UnitreeGo2FlatFlowPPORunnerCfg,
    "reflow": UnitreeGo2FlatFlowPPORunnerCfgReflow,
    "reward_aware": UnitreeGo2FlatFlowPPORunnerCfgRewardAware,
    "adaptive_compute": UnitreeGo2FlatFlowPPORunnerCfgAdaptiveCompute,
    "fpo_operator": UnitreeGo2FlatFlowPPORunnerCfgFpoOperator,
    "theory": UnitreeGo2FlatFlowPPORunnerCfgTheory,
}

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


# ---------------------------------------------------------------------------
# Humanoid locomotion (32 samples, 32 epochs, 2000 iters)
# ---------------------------------------------------------------------------


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

TASK_CONFIGS = {
    # Quadrupeds
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
