import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_robobenchmart_example() -> dict:
    """Creates a random input example for the RoboBenchMart policy."""
    return {
        "observation/state": np.random.rand(15),
        "observation/image": np.random.randint(256, size=(256, 256, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(256, 256, 3), dtype=np.uint8),
        "observation/extra_image": np.random.randint(256, size=(256, 256, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class RBMInputs(transforms.DataTransformFn):
    """Converts RoboBenchMart observations to the model input format."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])
        extra_image = _parse_image(data["observation/extra_image"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": extra_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class RBMOutputs(transforms.DataTransformFn):
    """Converts model action chunks back to RoboBenchMart actions."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :13])}
