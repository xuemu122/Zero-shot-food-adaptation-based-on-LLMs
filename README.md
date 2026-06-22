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
    "isaac_stage": "你的场景 USD 路径",
    "policy_checkpoint": "GRITS 或扩散策略 checkpoint 路径",
    "food_asset_root": "食物资产目录"
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
