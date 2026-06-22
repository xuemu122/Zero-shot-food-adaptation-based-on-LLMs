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
