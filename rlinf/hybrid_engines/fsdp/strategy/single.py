"""Single-device training strategy for models incompatible with FSDP wrapping."""

from __future__ import annotations

import os
from contextlib import nullcontext

import torch
import torch.nn as nn

from rlinf.hybrid_engines.fsdp.strategy.base import FSDPStrategyBase
from rlinf.hybrid_engines.fsdp.utils import FSDPVersion


class SingleDeviceStrategy(FSDPStrategyBase):
    def __init__(self, cfg, world_size, dp_group=None, logger=None):
        if world_size != 1:
            raise ValueError("The 'single' strategy requires world_size=1")
        super().__init__(cfg, world_size, dp_group, logger)

    def wrap_model(self, model: nn.Module, device_mesh):
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
        return model.to(device)

    @classmethod
    def get_fsdp_version(cls) -> FSDPVersion:
        # Required by the common interface; this strategy overrides checkpoint IO.
        return FSDPVersion.FSDP

    def clip_grad_norm_(self, model, norm_type=2.0):
        return torch.nn.utils.clip_grad_norm_(
            model.parameters(), self.cfg.optim.clip_grad, norm_type=norm_type
        )

    @classmethod
    def save_checkpoint(
        cls,
        model,
        optimizers,
        lr_schedulers,
        save_path,
        save_full_model_weights=True,
        checkpoint_format="dcp",
    ):
        del checkpoint_format
        os.makedirs(save_path, exist_ok=True)
        model_dir = os.path.join(save_path, "model_state_dict")
        os.makedirs(model_dir, exist_ok=True)
        state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save(state_dict, os.path.join(model_dir, "full_weights.pt"))
        torch.save(optimizers.state_dict(), os.path.join(save_path, "optimizer.pt"))
        torch.save(
            lr_schedulers.state_dict(), os.path.join(save_path, "lr_scheduler.pt")
        )

    @classmethod
    def load_checkpoint(
        cls,
        model,
        optimizers,
        lr_schedulers,
        load_path,
        checkpoint_format="dcp",
    ):
        del checkpoint_format
        model_path = os.path.join(load_path, "model_state_dict", "full_weights.pt")
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        optimizers.load_state_dict(
            torch.load(os.path.join(load_path, "optimizer.pt"), map_location="cpu")
        )
        lr_schedulers.load_state_dict(
            torch.load(os.path.join(load_path, "lr_scheduler.pt"), map_location="cpu")
        )

    def get_model_state_dict(self, model, cpu_offload, full_state_dict):
        del full_state_dict
        state = model.state_dict()
        if cpu_offload:
            return {k: v.detach().cpu() for k, v in state.items()}
        return state

    def load_model_with_state_dict(
        self, model, model_state_dict, cpu_offload, full_state_dict
    ):
        del cpu_offload, full_state_dict
        return model.load_state_dict(model_state_dict)

    def offload_optimizer(self, optimizer):
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.cpu()

    def onload_optimizer(self, optimizer, device):
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)

    def offload_param_and_grad(self, model, offload_grad):
        model.to("cpu")
        if offload_grad:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad = param.grad.cpu()

    def onload_param_and_grad(self, model, device, onload_grad):
        model.to(device)
        if onload_grad:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad = param.grad.to(device)

    def before_micro_batch(self, model, is_last_micro_batch):
        del model, is_last_micro_batch
        return nullcontext()
