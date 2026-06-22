from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from attribute_bias import BASE_ACTION, Food, load_foods, make_method_bias, train_bias_mapper


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class TrialResult:
    success: bool
    spill: bool
    trajectory_rmse_from_baseline: float
    notes: str = ""


class IsaacFoodScoopEnvAdapter:
    """Connect this adapter to the existing Isaac Sim food-scooping environment."""

    def __init__(self, simulation_app: Any, config: dict[str, Any]):
        self.simulation_app = simulation_app
        self.config = config

        # TODO: load your USD stage, robot, policy checkpoint, and food assets here.
        # Suggested config fields:
        # config["paths"]["isaac_stage"]
        # config["paths"]["policy_checkpoint"]
        # config["paths"]["food_asset_root"]

    def run_policy(
        self,
        food: Food,
        method: str,
        action_bias: np.ndarray,
        trial_seed: int,
    ) -> TrialResult:
        """Run one Isaac Sim trial and return metrics.

        Wire this method to your environment:
        1. Reset the scene and spawn `food.name`.
        2. Set food physical parameters from `food.attrs`.
        3. Run the original GRITS/diffusion policy for `baseline_grits`.
        4. For `llm_bias` and `human_upper_bound`, inject `action_bias` during denoising:
           A_{k-1} = A_{k-1} - rho * grad(J) + eta * action_bias.
        5. Detect success and spill from your existing task metrics.
        6. Compute RMSE between final biased trajectory and baseline trajectory if available.
        """
        raise NotImplementedError(
            "Connect IsaacFoodScoopEnvAdapter.run_policy to the existing Isaac Sim environment."
        )


class DryRunFoodScoopEnv:
    """Local smoke-test environment. Use IsaacFoodScoopEnvAdapter on the server."""

    def run_policy(
        self,
        food: Food,
        method: str,
        action_bias: np.ndarray,
        trial_seed: int,
    ) -> TrialResult:
        rng = random.Random(trial_seed)
        action = np.clip(BASE_ACTION + action_bias, 0.05, 0.95)
        target = np.clip(BASE_ACTION + _proxy_ideal_bias(food.attrs), 0.05, 0.95)
        distance = float(np.linalg.norm(action - target))
        hardness, stickiness, rollability, fragility = food.attrs
        difficulty = 0.22 + 0.15 * fragility + 0.10 * rollability + 0.08 * stickiness
        success_probability = 1.0 / (1.0 + math.exp(9.0 * (distance - difficulty)))
        success_probability = min(max(success_probability + rng.gauss(0.0, 0.035), 0.02), 0.98)
        speed_risk = max(0.0, action[0] - target[0]) + max(0.0, action[3] - target[3])
        angle_risk = abs(action[2] - target[2])
        spill_probability = 0.06 + 0.42 * speed_risk + 0.18 * angle_risk
        spill_probability += 0.20 * distance + 0.16 * rollability + 0.10 * stickiness + 0.12 * fragility
        spill_probability = min(max(spill_probability + rng.gauss(0.0, 0.025), 0.01), 0.95)
        spill = rng.random() < spill_probability
        success = rng.random() < success_probability
        if spill and rng.random() < 0.70:
            success = False
        rmse = float(np.sqrt(np.mean(action_bias**2)))
        return TrialResult(success=success, spill=spill, trajectory_rmse_from_baseline=rmse, notes="dry_run")


def _proxy_ideal_bias(attrs: np.ndarray) -> np.ndarray:
    from attribute_bias import ideal_bias

    return ideal_bias(attrs)


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["split"], row["method"]), []).append(row)
    summary: list[dict[str, Any]] = []
    for (split, method), items in sorted(grouped.items()):
        summary.append(
            {
                "split": split,
                "method": method,
                "trials": len(items),
                "success_rate": sum(int(x["success"]) for x in items) / len(items),
                "spill_rate": sum(int(x["spill"]) for x in items) / len(items),
                "trajectory_rmse_from_baseline": sum(float(x["trajectory_rmse_from_baseline"]) for x in items)
                / len(items),
            }
        )
    return summary


def launch_isaac(config: dict[str, Any]) -> Any:
    isaac_config = config.get("isaac", {})
    launch_config = {
        "headless": bool(isaac_config.get("headless", True)),
        "renderer": isaac_config.get("renderer", "RayTracedLighting"),
    }
    try:
        from isaacsim import SimulationApp
    except ImportError:
        from omni.isaac.kit import SimulationApp
    return SimulationApp(launch_config)


def run(config: dict[str, Any], dry_run: bool) -> tuple[Path, Path]:
    seed = int(config.get("seed", 42))
    rng = random.Random(seed)
    paths = config["paths"]
    foods = load_foods(resolve_path(paths["food_attributes_csv"]))
    train_foods = [food for food in foods if food.split == "train"]
    eval_foods = [food for food in foods if food.split in {"test", "real"}]
    weights = train_bias_mapper(train_foods)

    if dry_run:
        env: Any = DryRunFoodScoopEnv()
        simulation_app = None
    else:
        simulation_app = launch_isaac(config)
        env = IsaacFoodScoopEnvAdapter(simulation_app, config)

    rows: list[dict[str, Any]] = []
    methods = config["methods"]
    trials_by_split = config["trials"]
    bias_scale = float(config.get("bias_scale", 0.85))

    try:
        for food in eval_foods:
            trial_count = int(trials_by_split[food.split])
            for method in methods:
                for trial_index in range(trial_count):
                    trial_seed = rng.randint(0, 2**31 - 1)
                    action_bias, used_attrs = make_method_bias(method, food, weights, rng, bias_scale)
                    result = env.run_policy(food, method, action_bias, trial_seed)
                    rows.append(
                        {
                            "food": food.name,
                            "split": food.split,
                            "method": method,
                            "trial_index": trial_index,
                            "seed": trial_seed,
                            "success": int(result.success),
                            "spill": int(result.spill),
                            "trajectory_rmse_from_baseline": result.trajectory_rmse_from_baseline,
                            "bias_approach_speed": action_bias[0],
                            "bias_scoop_depth": action_bias[1],
                            "bias_tilt_angle": action_bias[2],
                            "bias_lift_speed": action_bias[3],
                            "attr_hardness": used_attrs[0],
                            "attr_stickiness": used_attrs[1],
                            "attr_rollability": used_attrs[2],
                            "attr_fragility": used_attrs[3],
                            "notes": result.notes,
                        }
                    )
    finally:
        if simulation_app is not None:
            simulation_app.close()

    output_csv = resolve_path(paths["output_csv"])
    summary_csv = resolve_path(paths["summary_csv"])
    write_csv(output_csv, rows)
    write_csv(summary_csv, summarize(rows))
    return output_csv, summary_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "isaacsim_experiment.json"))
    parser.add_argument("--dry-run", action="store_true", help="Run without Isaac Sim for a pipeline smoke test.")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    output_csv, summary_csv = run(config, dry_run=args.dry_run)
    print(f"Wrote trial results: {output_csv}")
    print(f"Wrote summary: {summary_csv}")


if __name__ == "__main__":
    main()
