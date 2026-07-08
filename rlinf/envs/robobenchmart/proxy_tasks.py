# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import pandas as pd

from mani_skill.utils.registration import register_env

from dsynth.envs.pick_to_basket import PickToBasketContEnv, PICK_TO_BASKET_DOC_STRING


EXCLUDED_PRODUCT_NAMES = (
    "Fanta Sabor Naranja 2L",
    "Nivea Body Milk",
    "Nestle Honey Stars",
    "Nestle Fitness Chocolate Cereals",
    "SLAM luncheon meat",
    "Duff Beer Can",
    "Vanish Stain Remover",
)


@register_env("PickToBasketProxyRandomEnv", max_episode_steps=200000)
class PickToBasketProxyRandomEnv(PickToBasketContEnv):
    """PickToBasket proxy task that excludes held-out OOD targets.

    This task samples one target product per scene from proxy items that are
    present in the training PickToBasket scenes. It deliberately excludes the
    final OOD evaluation targets Nestle/Slam/Duff and the three SFT train
    targets Fanta/Nivea/Stars.
    """

    TARGET_PRODUCT_NAME = None
    EXCLUDED_PRODUCT_NAMES = EXCLUDED_PRODUCT_NAMES

    def setup_target_objects(self, env_idxs):
        self.target_product_names = {}
        self.target_products_df = None

        if self.markers_enabled:
            target_markers_iterator = {
                key: iter(val) for key, val in self.target_markers.items()
            }

        for scene_idx in env_idxs:
            scene_idx = scene_idx.cpu().item()
            scene_products_df = self.products_df[self.products_df["scene_idx"] == scene_idx]
            present_proxy_products = sorted(
                set(scene_products_df["product_name"].unique())
                - set(self.EXCLUDED_PRODUCT_NAMES)
            )
            if not present_proxy_products:
                raise RuntimeError(
                    f"No proxy products are present on scene #{scene_idx}. "
                    f"Excluded products: {self.EXCLUDED_PRODUCT_NAMES}"
                )

            product_name = self._batched_episode_rng[scene_idx].choice(
                present_proxy_products
            )
            self.target_product_names[scene_idx] = product_name
            scene_target_products_df = scene_products_df[
                scene_products_df["product_name"] == product_name
            ]

            if self.target_products_df is None:
                self.target_products_df = scene_target_products_df
            else:
                self.target_products_df = pd.concat(
                    [self.target_products_df, scene_target_products_df]
                )

            if self.markers_enabled:
                for actor_name in scene_target_products_df["actor_name"]:
                    actor = self.actors["products"][actor_name]
                    try:
                        target_marker = next(target_markers_iterator[scene_idx])
                    except StopIteration as exc:
                        raise RuntimeError(
                            "Number of target objects exceeds number of markers "
                            f"({self.NUM_MARKERS}) for scene #{scene_idx}"
                        ) from exc
                    target_marker.set_pose(actor.pose)


PickToBasketProxyRandomEnv.__doc__ = PICK_TO_BASKET_DOC_STRING.format(
    product_name="a proxy product"
)
