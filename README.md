# LLM Zero-Shot Food Adaptation Simulation

这个目录已经搭好一个可运行的“基于 LLM 的零样本食物适应”仿真版本，对应 `shiyanshuoming` 里的实验思路。

## 仿真设置

仿真把真实舀取动作简化为 4 维动作空间：

- approach speed：进勺速度
- scoop depth：舀取深度
- tilt angle：倾斜角
- lift speed：抬升速度

每种食物用 4 维物理属性表示：

- hardness：硬度
- stickiness：粘性
- rollability：易滚动度
- fragility：易碎性

数据设置：

- 5 种合成食物用于训练属性到动作偏置的映射。
- 10 种合成食物用于零样本测试，每种 30 次试验。
- 10 种真实食物名称作为真实环境代理集，每种 5 次试验。
- 对比方法包括 baseline GRITS、随机偏置、LLM 决策提示、LLM 偏置、人工上界。

## 运行

使用 Codex 内置 Python：

```powershell
& 'C:\Users\P4-543\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' .\src\run_simulation.py
```

输出会写入 `outputs/`：

- `summary.csv`：最重要的汇总结果。
- `experiment_results.csv`：每种食物、每种方法的详细结果。
- `attribute_mse.csv`：LLM 属性预测误差。
- `test_success_rate.svg`、`test_spill_rate.svg`：零样本模拟结果图。
- `real_success_rate.svg`、`real_spill_rate.svg`：真实食物代理集结果图。

当前一次运行的关键结果：

- 零样本 baseline 成功率：0.600
- 零样本 LLM 偏置成功率：0.783
- 零样本人工上界成功率：0.837
- baseline 溢漏率：0.333
- LLM 偏置溢漏率：0.230

## Isaac Sim 版本

如果使用 Isaac Sim，在服务器上优先跑这个入口：

```bash
./python.sh /path/to/project/src/isaacsim_runner.py --config /path/to/project/configs/isaacsim_experiment.json
```

如果服务器上的 Isaac Sim Python 命令不是 `./python.sh`，就换成你们安装目录里的 Isaac Sim Python，例如：

```bash
/isaac-sim/python.sh /path/to/project/src/isaacsim_runner.py --config /path/to/project/configs/isaacsim_experiment.json
```

本地检查实验流程可以先跑 dry-run：

```powershell
& 'C:\Users\P4-543\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' .\src\isaacsim_runner.py --dry-run
```

服务器真正接 Isaac Sim 时，主要改两个地方：

1. 在 `configs/isaacsim_experiment.json` 里填路径：

```json
{
  "paths": {
    "isaac_stage": "dafan/dafan/dafanchangjing.usd",
    "robot_urdf": "dafan/dafan/local_mqon69sb_ywdvm7_urdf_stl/robot.urdf",
    "policy_checkpoint": "GRITS 或扩散策略 checkpoint 路径",
    "food_asset_root": "dafan/dafan/food",
    "isaac_food_assets_csv": "data/isaac_food_assets.csv"
  }
}
```

2. 在 `src/isaacsim_runner.py` 里实现 `IsaacFoodScoopEnvAdapter.run_policy(...)`。

这个函数每次会收到：

- `food`：当前食物，包含名称和 4 维属性。
- `method`：当前方法，例如 `baseline_grits`、`llm_bias`。
- `action_bias`：4 维动作偏置，对应进勺速度、舀取深度、倾斜角、抬升速度。
- `trial_seed`：本次试验随机种子。

你们环境里需要做的是：

- reset 场景并加载当前食物；
- 根据 `food.attrs` 设置物理参数，或选择对应资产；
- baseline 方法不加偏置；
- `llm_bias` 和 `human_upper_bound` 在扩散去噪时注入偏置；
- 返回成功、是否溢漏、轨迹相对 baseline 的 RMSE。

偏置注入建议放在扩散策略每一步去噪更新处：

```text
A_{k-1} = A_{k-1} - rho * grad(J) + eta * action_bias
```

跑完后会生成：

- `outputs/isaacsim_experiment_results.csv`
- `outputs/isaacsim_summary.csv`

## 当前资产检查结论

- 推荐使用主场景：`dafan/dafan/dafanchangjing.usd`。
- 推荐使用机器人：`dafan/dafan/local_mqon69sb_ywdvm7_urdf_stl/robot.urdf`，这个 URDF 的 mesh 引用完整，并且替换了旧版关节方向有问题的机械臂。
- 旧版 `dafan/dafan/local_mqdeszw7_2fp5fc_urdf_stl（jiu）/robot.urdf` 已标注为旧，不建议继续用于实验。
- 暂时不要直接用 `dafan/dafan/so101_new_calib.urdf`，它引用的 `assets/*.stl` 当前目录里没有。
- 食物资产在 `dafan/dafan/food/`，包括片、条、块、球等形状。
- 食物 USD 里大多已有 RigidBody，但没有看到明确的 mass/friction/restitution 字段；建议运行时根据 `data/isaac_food_assets.csv` 给每个食物补物理材质。

## 推荐物理参数

Isaac Sim 里建议至少给每个食物设置这些参数：

- `mass`：质量，影响舀取时是否容易被推走。
- `static_friction`：静摩擦，影响食物起步滑动难度。
- `dynamic_friction`：动摩擦，影响滑动过程中的阻力。
- `restitution`：恢复系数，影响碰撞后反弹程度。
- `linear_damping`：线速度阻尼，可抑制不真实滑动。
- `angular_damping`：角速度阻尼，可抑制不真实滚动和翻转。

推荐初始范围：

| 形状 | mass | static_friction | dynamic_friction | restitution | linear_damping | angular_damping | 说明 |
|---|---:|---:|---:|---:|---:|---:|---|
| 薄片 / 片状 | 0.015-0.035 | 0.7-1.3 | 0.5-1.0 | 0.0-0.08 | 0.05-0.15 | 0.10-0.30 | 容易翻、贴勺边，适合模拟粘性和易碎性 |
| 条状 | 0.035-0.070 | 0.25-0.60 | 0.18-0.45 | 0.10-0.25 | 0.03-0.10 | 0.05-0.20 | 有方向性，圆条易滚，扁条可降低 rollability |
| 小块 | 0.040-0.070 | 0.45-0.80 | 0.30-0.60 | 0.03-0.15 | 0.02-0.08 | 0.03-0.12 | 作为中等难度样本，比较稳定 |
| 大块 | 0.080-0.130 | 0.50-0.90 | 0.35-0.70 | 0.02-0.12 | 0.03-0.10 | 0.04-0.15 | 质量更大，容易考察进勺深度和力控制 |
| 球 / 圆粒 | 0.025-0.060 | 0.10-0.35 | 0.06-0.25 | 0.25-0.60 | 0.01-0.05 | 0.01-0.08 | 主要用于测试易滚动和溢漏 |

当前食物可以先按下面这组参数设：

| 食物 | 类型 | mass | static_friction | dynamic_friction | restitution | hardness | stickiness | rollability | fragility |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `bopian.usd` | 薄片 | 0.018 | 1.10 | 0.85 | 0.02 | 0.28 | 0.72 | 0.15 | 0.75 |
| `pian.usd` | 片 | 0.026 | 0.75 | 0.55 | 0.05 | 0.38 | 0.45 | 0.20 | 0.58 |
| `tiao(ying).usd` | 条 | 0.045 | 0.35 | 0.25 | 0.18 | 0.70 | 0.15 | 0.72 | 0.25 |
| `qiu.usd` | 球 | 0.035 | 0.22 | 0.16 | 0.35 | 0.55 | 0.08 | 0.92 | 0.12 |
| `xiaokeli.usd` | 小块 | 0.050 | 0.55 | 0.42 | 0.08 | 0.62 | 0.25 | 0.35 | 0.25 |
| `dakeli.usd` | 大块 | 0.095 | 0.62 | 0.48 | 0.06 | 0.68 | 0.22 | 0.25 | 0.20 |

调参时优先保证现象合理：

- 如果食物一碰就乱飞：降低 `restitution`，提高 `linear_damping`。
- 如果球不滚：降低摩擦，降低 `angular_damping`。
- 如果片状食物完全不滑：降低 `static_friction` 和 `dynamic_friction`。
- 如果粘性食物表现不出来：提高摩擦，降低 `restitution`，适当提高阻尼。
- 如果所有食物都太容易舀起：增加质量或降低摩擦差异，让形状和接触更有影响。
