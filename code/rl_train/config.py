"""Configuration container for PPO training."""

import math
from dataclasses import dataclass


@dataclass
class PPOConfig:
    """
    Hyperparameters and runtime options for PPO training.

    Attributes:
        obs_dim: Dimension of observation vector fed to policy/value networks.
        theta_dim: Dimension of policy output parameter vector (theta_t).
        max_steps: Per-episode hard horizon used by the environment wrapper.
        total_timesteps: Total environment interaction budget.
        rollout_steps: Number of transitions collected before each PPO update.
        update_epochs: Number of optimization passes over each rollout batch.
        minibatch_size: Minibatch size inside each PPO update epoch.
        learning_rate: Adam learning rate.
        gamma: Discount factor for future rewards.
        gae_lambda: Lambda parameter for generalized advantage estimation.
        clip_eps: PPO clipping epsilon for policy/value updates.
        ent_coef: Entropy coefficient for exploration encouragement.
        vf_coef: Value loss coefficient.
        max_grad_norm: Gradient clipping threshold.
        hidden_dim: Width of hidden layers in actor and critic MLPs.
        use_layer_norm: Whether to insert LayerNorm after hidden linear layers.
        seed: Random seed for NumPy/PyTorch/Python RNG.
        device: Torch device string, for example "cpu" or "cuda".
    """

    # Core problem size
    obs_dim: int = 16
    theta_dim: int = 4
    max_steps: int = 50

    # Training
    total_timesteps: int = 200_000
    rollout_steps: int = 256
    update_epochs: int = 4
    minibatch_size: int = 128
    learning_rate: float = 3e-4

    # PPO losses
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Network / reproducibility
    hidden_dim: int = 128
    use_layer_norm: bool = False
    seed: int = 42
    device: str = "cpu"

    def validate(self) -> None:
        """Raise a readable error when PPO settings cannot produce a valid run."""
        errors: list[str] = []
        for name in (
            "obs_dim",
            "theta_dim",
            "max_steps",
            "total_timesteps",
            "rollout_steps",
            "update_epochs",
            "minibatch_size",
            "hidden_dim",
        ):
            if getattr(self, name) <= 0:
                errors.append(f"{name} must be greater than 0 (got {getattr(self, name)!r})")

        for name in ("learning_rate", "max_grad_norm"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                errors.append(f"{name} must be a finite value greater than 0 (got {value!r})")

        for name in ("gamma", "gae_lambda"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                errors.append(f"{name} must be between 0 and 1 inclusive (got {value!r})")

        for name in ("clip_eps", "ent_coef", "vf_coef"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                errors.append(f"{name} must be a finite non-negative value (got {value!r})")

        if not isinstance(self.device, str) or not self.device.strip():
            errors.append("device must be a non-empty Torch device string")

        if errors:
            raise ValueError("Invalid PPO configuration:\n  - " + "\n  - ".join(errors))
