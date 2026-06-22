# Simulation Report

This is a lightweight simulation for the experiment idea: convert food physical attributes into a trajectory/action bias for zero-shot scooping.

## Setup

- Food attributes: hardness, stickiness, rollability, fragility.
- Action vector: approach speed, scoop depth, tilt angle, lift speed.
- Training foods: 5 synthetic foods.
- Zero-shot simulation foods: 10 synthetic foods, 30 trials each.
- Real-food proxy set: 10 named foods, 5 trials each.
- Compared methods: baseline GRITS, random bias, LLM decision only, LLM bias, human upper bound.

## Key Simulated Result

- Baseline success rate on synthetic zero-shot foods: 0.600
- LLM-bias success rate on synthetic zero-shot foods: 0.783
- Absolute improvement: 0.183
- Mean synthetic attribute MSE: 0.0058

## Files

- `experiment_results.csv`: per-food metrics.
- `summary.csv`: averaged metrics by split and method.
- `attribute_mse.csv`: noisy LLM-attribute proxy versus known simulator attributes.
- `test_success_rate.svg`, `test_spill_rate.svg`, `real_success_rate.svg`, `real_spill_rate.svg`: simple charts.

## Interpretation

This simulation is not a physics engine. It is a controlled sandbox for checking whether the experiment pipeline is coherent before connecting real GRITS code, a robot, or a particle simulator.
