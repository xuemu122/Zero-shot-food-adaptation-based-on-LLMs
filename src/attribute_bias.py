from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np


BASE_ACTION = np.array([0.50, 0.50, 0.50, 0.50])


@dataclass(frozen=True)
class Food:
    name: str
    split: str
    source: str
    attrs: np.ndarray
    asset_path: str = ""
    shape: str = ""
    mass: float = 0.0
    static_friction: float = 0.0
    dynamic_friction: float = 0.0
    restitution: float = 0.0


def load_foods(path: Path) -> list[Food]:
    foods: list[Food] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            attrs = np.array(
                [
                    float(row["hardness"]),
                    float(row["stickiness"]),
                    float(row["rollability"]),
                    float(row["fragility"]),
                ],
                dtype=float,
            )
            foods.append(Food(row["food"], row["split"], row["source"], attrs))
    return foods


def load_isaac_food_assets(path: Path) -> list[Food]:
    foods: list[Food] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            attrs = np.array(
                [
                    float(row["hardness"]),
                    float(row["stickiness"]),
                    float(row["rollability"]),
                    float(row["fragility"]),
                ],
                dtype=float,
            )
            foods.append(
                Food(
                    name=row["food"],
                    split=row["split"],
                    source="isaac_asset",
                    attrs=attrs,
                    asset_path=row["asset_path"],
                    shape=row["shape"],
                    mass=float(row["mass"]),
                    static_friction=float(row["static_friction"]),
                    dynamic_friction=float(row["dynamic_friction"]),
                    restitution=float(row["restitution"]),
                )
            )
    return foods


def ideal_action(attrs: np.ndarray) -> np.ndarray:
    """Proxy safe action used until Isaac Sim safe/unsafe trajectories are available."""
    hardness, stickiness, rollability, fragility = attrs
    action = np.array(
        [
            0.62 - 0.26 * fragility - 0.10 * rollability + 0.08 * hardness,
            0.46 + 0.24 * hardness + 0.12 * stickiness - 0.16 * fragility,
            0.48 + 0.22 * rollability + 0.10 * hardness - 0.16 * stickiness,
            0.58 - 0.24 * fragility - 0.14 * stickiness + 0.08 * hardness,
        ]
    )
    return np.clip(action, 0.05, 0.95)


def ideal_bias(attrs: np.ndarray) -> np.ndarray:
    return ideal_action(attrs) - BASE_ACTION


def train_bias_mapper(train_foods: list[Food]) -> np.ndarray:
    x = np.vstack([np.append(food.attrs, 1.0) for food in train_foods])
    y = np.vstack([ideal_bias(food.attrs) for food in train_foods])
    ridge = 1e-3 * np.eye(x.shape[1])
    return np.linalg.solve(x.T @ x + ridge, x.T @ y)


def predict_bias(attrs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    features = np.append(attrs, 1.0)
    return np.clip(features @ weights, -0.35, 0.35)


def noisy_llm_attrs(true_attrs: np.ndarray, rng: random.Random) -> np.ndarray:
    noise = np.array([rng.gauss(0.0, 0.08) for _ in range(4)])
    return np.clip(true_attrs + noise, 0.0, 1.0)


def make_method_bias(
    method: str,
    food: Food,
    weights: np.ndarray,
    rng: random.Random,
    bias_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return action-space bias and the attribute vector used to derive it."""
    if method == "baseline_grits":
        return np.zeros(4), food.attrs
    if method == "random_bias":
        return np.array([rng.uniform(-0.22, 0.22) for _ in range(4)]), food.attrs
    if method == "llm_bias":
        attrs = noisy_llm_attrs(food.attrs, rng)
        return bias_scale * predict_bias(attrs, weights), attrs
    if method == "human_upper_bound":
        return ideal_bias(food.attrs), food.attrs
    if method == "llm_decision_only":
        attrs = noisy_llm_attrs(food.attrs, rng)
        cautious = 0.08 * attrs[3] + 0.05 * attrs[1]
        return np.array([-cautious, 0.0, 0.0, -cautious]), attrs
    raise ValueError(f"Unknown method: {method}")
