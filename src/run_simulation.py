from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "foods.csv"
OUTPUT_DIR = ROOT / "outputs"
TRIALS_PER_SIM_FOOD = 30
TRIALS_PER_REAL_FOOD = 5
RANDOM_SEED = 42

# Action vector: [approach_speed, scoop_depth, tilt_angle, lift_speed].
# A neutral value of 0.5 means the unadapted baseline action.
BASE_ACTION = np.array([0.50, 0.50, 0.50, 0.50])
BIAS_SCALE = 0.85


@dataclass(frozen=True)
class Food:
    name: str
    split: str
    source: str
    attrs: np.ndarray


def load_foods() -> list[Food]:
    foods: list[Food] = []
    with DATA_PATH.open("r", encoding="utf-8", newline="") as f:
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


def ideal_action(attrs: np.ndarray) -> np.ndarray:
    """Ground-truth simulator policy used to define safe scooping."""
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
    # Small ridge term keeps the mapping stable with only five training foods.
    ridge = 1e-3 * np.eye(x.shape[1])
    return np.linalg.solve(x.T @ x + ridge, x.T @ y)


def predict_bias(attrs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    features = np.append(attrs, 1.0)
    return np.clip(features @ weights, -0.35, 0.35)


def noisy_llm_attrs(true_attrs: np.ndarray, rng: random.Random) -> np.ndarray:
    noise = np.array([rng.gauss(0.0, 0.08) for _ in range(4)])
    return np.clip(true_attrs + noise, 0.0, 1.0)


def simulate_trial(food: Food, action: np.ndarray, rng: random.Random) -> tuple[bool, bool, float]:
    target = ideal_action(food.attrs)
    distance = float(np.linalg.norm(action - target))
    hardness, stickiness, rollability, fragility = food.attrs

    difficulty = 0.22 + 0.15 * fragility + 0.10 * rollability + 0.08 * stickiness
    success_probability = 1.0 / (1.0 + math.exp(9.0 * (distance - difficulty)))
    success_probability += rng.gauss(0.0, 0.035)
    success_probability = min(max(success_probability, 0.02), 0.98)
    success = rng.random() < success_probability

    speed_risk = max(0.0, action[0] - target[0]) + max(0.0, action[3] - target[3])
    angle_risk = abs(action[2] - target[2])
    spill_probability = 0.06 + 0.42 * speed_risk + 0.18 * angle_risk
    spill_probability += 0.20 * distance
    spill_probability += 0.16 * rollability + 0.10 * stickiness + 0.12 * fragility
    spill_probability += rng.gauss(0.0, 0.025)
    spill_probability = min(max(spill_probability, 0.01), 0.95)
    spill = rng.random() < spill_probability

    if spill and rng.random() < 0.70:
        success = False
    return success, spill, distance


def method_action(
    method: str,
    food: Food,
    weights: np.ndarray,
    rng: random.Random,
) -> tuple[np.ndarray, float]:
    if method == "baseline_grits":
        action = BASE_ACTION.copy()
    elif method == "random_bias":
        random_bias = np.array([rng.uniform(-0.22, 0.22) for _ in range(4)])
        action = BASE_ACTION + random_bias
    elif method == "llm_bias":
        attrs = noisy_llm_attrs(food.attrs, rng)
        action = BASE_ACTION + BIAS_SCALE * predict_bias(attrs, weights)
    elif method == "human_upper_bound":
        action = ideal_action(food.attrs)
    elif method == "llm_decision_only":
        attrs = noisy_llm_attrs(food.attrs, rng)
        cautious = 0.08 * attrs[3] + 0.05 * attrs[1]
        action = BASE_ACTION + np.array([-cautious, 0.0, 0.0, -cautious])
    else:
        raise ValueError(f"Unknown method: {method}")
    action = np.clip(action, 0.05, 0.95)
    rmse = float(np.sqrt(np.mean((action - BASE_ACTION) ** 2)))
    return action, rmse


def run_experiments(foods: list[Food], weights: np.ndarray) -> list[dict[str, str | float | int]]:
    rng = random.Random(RANDOM_SEED)
    methods = [
        "baseline_grits",
        "random_bias",
        "llm_decision_only",
        "llm_bias",
        "human_upper_bound",
    ]
    rows: list[dict[str, str | float | int]] = []
    eval_foods = [food for food in foods if food.split in {"test", "real"}]

    for food in eval_foods:
        trials = TRIALS_PER_REAL_FOOD if food.split == "real" else TRIALS_PER_SIM_FOOD
        for method in methods:
            successes = 0
            spills = 0
            distances: list[float] = []
            rmses: list[float] = []
            for _ in range(trials):
                action, rmse = method_action(method, food, weights, rng)
                success, spill, distance = simulate_trial(food, action, rng)
                successes += int(success)
                spills += int(spill)
                distances.append(distance)
                rmses.append(rmse)
            rows.append(
                {
                    "food": food.name,
                    "split": food.split,
                    "method": method,
                    "trials": trials,
                    "success_rate": successes / trials,
                    "spill_rate": spills / trials,
                    "mean_action_distance": sum(distances) / trials,
                    "trajectory_rmse_from_baseline": sum(rmses) / trials,
                }
            )
    return rows


def attribute_mse_rows(foods: list[Food]) -> list[dict[str, str | float]]:
    rng = random.Random(RANDOM_SEED + 7)
    rows: list[dict[str, str | float]] = []
    for food in foods:
        if food.split != "test":
            continue
        pred = noisy_llm_attrs(food.attrs, rng)
        mse = float(np.mean((pred - food.attrs) ** 2))
        rows.append(
            {
                "food": food.name,
                "hardness_true": food.attrs[0],
                "stickiness_true": food.attrs[1],
                "rollability_true": food.attrs[2],
                "fragility_true": food.attrs[3],
                "hardness_llm": pred[0],
                "stickiness_llm": pred[1],
                "rollability_llm": pred[2],
                "fragility_llm": pred[3],
                "attribute_mse": mse,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, str | float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, str | float | int]]) -> list[dict[str, str | float]]:
    grouped: dict[tuple[str, str], list[dict[str, str | float | int]]] = {}
    for row in rows:
        key = (str(row["split"]), str(row["method"]))
        grouped.setdefault(key, []).append(row)

    out: list[dict[str, str | float]] = []
    for (split, method), items in sorted(grouped.items()):
        out.append(
            {
                "split": split,
                "method": method,
                "success_rate": sum(float(x["success_rate"]) for x in items) / len(items),
                "spill_rate": sum(float(x["spill_rate"]) for x in items) / len(items),
                "mean_action_distance": sum(float(x["mean_action_distance"]) for x in items)
                / len(items),
                "trajectory_rmse_from_baseline": sum(
                    float(x["trajectory_rmse_from_baseline"]) for x in items
                )
                / len(items),
            }
        )
    return out


def write_svg_bar_chart(path: Path, rows: list[dict[str, str | float]], split: str, metric: str) -> None:
    items = [row for row in rows if row["split"] == split]
    labels = [str(row["method"]) for row in items]
    values = [float(row[metric]) for row in items]
    width = 860
    height = 430
    left = 170
    top = 40
    bar_h = 42
    gap = 25
    max_value = max(1.0, max(values))
    colors = ["#4b5563", "#9ca3af", "#60a5fa", "#22c55e", "#f59e0b"]
    title = f"{split} {metric}"

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="26" font-family="Arial" font-size="20" fill="#111827">{title}</text>',
    ]
    for i, (label, value) in enumerate(zip(labels, values)):
        y = top + i * (bar_h + gap)
        bar_w = int((width - left - 90) * value / max_value)
        lines.append(
            f'<text x="18" y="{y + 28}" font-family="Arial" font-size="14" fill="#111827">{label}</text>'
        )
        lines.append(
            f'<rect x="{left}" y="{y}" width="{bar_w}" height="{bar_h}" fill="{colors[i % len(colors)]}"/>'
        )
        lines.append(
            f'<text x="{left + bar_w + 10}" y="{y + 28}" font-family="Arial" font-size="14" fill="#111827">{value:.2f}</text>'
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_readme(summary_rows: list[dict[str, str | float]], attr_rows: list[dict[str, str | float]]) -> None:
    sim_llm = next(row for row in summary_rows if row["split"] == "test" and row["method"] == "llm_bias")
    sim_base = next(row for row in summary_rows if row["split"] == "test" and row["method"] == "baseline_grits")
    improvement = float(sim_llm["success_rate"]) - float(sim_base["success_rate"])
    avg_mse = sum(float(row["attribute_mse"]) for row in attr_rows) / len(attr_rows)
    text = f"""# Simulation Report

This is a lightweight simulation for the experiment idea: convert food physical attributes into a trajectory/action bias for zero-shot scooping.

## Setup

- Food attributes: hardness, stickiness, rollability, fragility.
- Action vector: approach speed, scoop depth, tilt angle, lift speed.
- Training foods: 5 synthetic foods.
- Zero-shot simulation foods: 10 synthetic foods, 30 trials each.
- Real-food proxy set: 10 named foods, 5 trials each.
- Compared methods: baseline GRITS, random bias, LLM decision only, LLM bias, human upper bound.

## Key Simulated Result

- Baseline success rate on synthetic zero-shot foods: {float(sim_base["success_rate"]):.3f}
- LLM-bias success rate on synthetic zero-shot foods: {float(sim_llm["success_rate"]):.3f}
- Absolute improvement: {improvement:.3f}
- Mean synthetic attribute MSE: {avg_mse:.4f}

## Files

- `experiment_results.csv`: per-food metrics.
- `summary.csv`: averaged metrics by split and method.
- `attribute_mse.csv`: noisy LLM-attribute proxy versus known simulator attributes.
- `test_success_rate.svg`, `test_spill_rate.svg`, `real_success_rate.svg`, `real_spill_rate.svg`: simple charts.

## Interpretation

This simulation is not a physics engine. It is a controlled sandbox for checking whether the experiment pipeline is coherent before connecting real GRITS code, a robot, or a particle simulator.
"""
    (OUTPUT_DIR / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    foods = load_foods()
    train_foods = [food for food in foods if food.split == "train"]
    weights = train_bias_mapper(train_foods)

    rows = run_experiments(foods, weights)
    summary = aggregate(rows)
    attr_rows = attribute_mse_rows(foods)

    write_csv(OUTPUT_DIR / "experiment_results.csv", rows)
    write_csv(OUTPUT_DIR / "summary.csv", summary)
    write_csv(OUTPUT_DIR / "attribute_mse.csv", attr_rows)
    np.savetxt(OUTPUT_DIR / "bias_mapper_weights.csv", weights, delimiter=",")

    write_svg_bar_chart(OUTPUT_DIR / "test_success_rate.svg", summary, "test", "success_rate")
    write_svg_bar_chart(OUTPUT_DIR / "test_spill_rate.svg", summary, "test", "spill_rate")
    write_svg_bar_chart(OUTPUT_DIR / "real_success_rate.svg", summary, "real", "success_rate")
    write_svg_bar_chart(OUTPUT_DIR / "real_spill_rate.svg", summary, "real", "spill_rate")
    write_readme(summary, attr_rows)

    print("Simulation complete.")
    print(f"Wrote outputs to: {OUTPUT_DIR}")
    for row in summary:
        print(
            f"{row['split']:>4} | {row['method']:<18} "
            f"success={float(row['success_rate']):.3f} "
            f"spill={float(row['spill_rate']):.3f} "
            f"rmse={float(row['trajectory_rmse_from_baseline']):.3f}"
        )


if __name__ == "__main__":
    main()
