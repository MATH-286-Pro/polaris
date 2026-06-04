from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent
    / "2026-05-27_02-22-04"
    / "exported"
    / "policy_10009.jit"
)


class RSLPolicy:
    """Thin wrapper around an exported RSL torchscript policy."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        device: str | torch.device | None = None,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.policy = torch.jit.load(str(self.model_path), map_location=self.device)
        self.policy.eval()
        self.obs_dim = self.infer_obs_dim(self.policy)

    # no need for self, 
    # so we can use @staticmethod
    @staticmethod
    def infer_obs_dim(policy: torch.jit.ScriptModule) -> int:
        state_dict = policy.state_dict()

        # If the exported policy has a normalizer, its 1D buffers match obs_dim.
        for name, value in state_dict.items():
            if name.startswith("normalizer.") and value.ndim == 1:
                return value.numel()

        # Recurrent policy: first RNN input weight is [gates * hidden, obs_dim].
        for name, value in state_dict.items():
            if name.startswith("rnn.") and "weight_ih_l0" in name:
                return value.shape[1]

        # Feed-forward policy: first actor Linear weight is [hidden_dim, obs_dim].
        for name, value in state_dict.items():
            if name.startswith("actor.") and name.endswith(".weight") and value.ndim == 2:
                return value.shape[1]

        raise RuntimeError("Could not infer obs_dim from policy.")

    def infer(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        if obs.shape[-1] != self.obs_dim:
            raise ValueError(f"Expected obs_dim={self.obs_dim}, got {obs.shape[-1]}.")

        with torch.inference_mode():
            return self.policy(obs.to(device=self.device))

    def reset(self) -> None:
        reset_fn = getattr(self.policy, "reset", None)
        if callable(reset_fn):
            reset_fn()

    @torch.inference_mode()
    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        return self.infer(obs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.policy, name)

# Use Example
if __name__ == "__main__":
    policy = RSLPolicy()
    obs = torch.zeros(1, policy.obs_dim)
    action = policy(obs)

    print(policy.obs_dim)
    print(action.shape)
