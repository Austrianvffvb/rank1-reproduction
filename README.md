# Rank1 小规模微调与推理文本监督消融

基于 **Qwen2.5-1.5B Base**，复现 Rank1 的数据处理与监督微调流程，探索在有限数据下，推理文本监督能否改善查询—段落相关性判断。

**范围：这是相关性二分类验证，不是完整的 Rank1 重排序基准复现。** 本项目未运行 BRIGHT、未实现官方概率评分排序、未报告 nDCG@10，也未进行模型内部机制或推理忠实性干预。实验于 2026 年 10 月在单卡云 GPU 上运行。

- 原论文：[Rank1: Test-Time Compute for Reranking in Information Retrieval](https://arxiv.org/abs/2502.18418)
- 官方实现：[orionw/rank1](https://github.com/orionw/rank1)
- 官方训练数据：[jhu-clsp/rank1-training-data](https://huggingface.co/datasets/jhu-clsp/rank1-training-data)
- 基础模型：[Qwen/Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B)

## 研究问题与实验设计

比较三个模型：原始 Base、输出“分析文本＋标签”的微调模型、仅输出标签的微调模型。两个微调组使用相同的 5000 条输入、主要超参数和参数训练范围，均从基础模型开始；仅标签组去掉输出中的分析文本。

原始文件含 385,791 条样本；精确去重后剩 338,140 条。按归一化查询分组，先将 10% 唯一查询划入验证池，再固定随机种子抽样，获得：

| 划分 | 样本对 | 不同查询 | true / false |
|---|---:|---:|---:|
| 训练 | 5,000 | 4,954 | 2,012 / 2,988 |
| 验证 | 500 | 495 | 208 / 292 |

训练和验证的归一化查询交集为 0。验证集内部允许一个查询对应多个段落。详见 [数据清单](data/data_manifest.json)。这是从官方训练数据中另外划出的验证集，不是官方测试集。

## 实验结果

由归档的逐样本预测重新计算，非新一轮模型推理。无效标签按错误计入总体准确率。

| 模型 | 正确数 | 准确率 | 按查询 bootstrap 的 95% 区间 | 有效标签率 | 平均生成 token |
|---|---:|---:|---:|---:|---:|
| 原始 Base | 353/500 | 70.6% | [66.7%, 74.6%] | 98.8% | 7.344 |
| 推理文本监督 | 446/500 | 89.2% | [86.4%, 92.0%] | 100% | 203.808 |
| 仅标签监督 | 453/500 | 90.6% | [88.1%, 93.0%] | 100% | 1.000 |

- 推理组较 Base 高 **18.6 个百分点**；仅标签组较推理组高 **1.4 个百分点**。
- 两组都对 422 条；仅推理组对 24 条；仅标签组对 31 条；两组都错 23 条。
- 按归一化查询进行 2,000 次配对 bootstrap，种子 12345；估计量是样本对加权准确率。仅标签减推理组差值的 95% 百分位区间约 **[-1.59, 4.24] 个百分点**，跨过 0。
- 这是验证集条件下的探索性区间，不覆盖训练种子波动、数据选择或验证集使用带来的不确定性，不能证明两种方法等价或推理一般无效。
- 既有单样本耗时来自不同 GPU，不计算跨组加速倍数。

来源：[逐样本预测](results/predictions/)、[重算结果](results/verified_summary.json)、[统计与结论边界](docs/limitations.md)。

## 运行步骤

### 1. 环境

原实验为 Linux、Python 3.12、CUDA PyTorch 2.8.0+cu128、LLaMA-Factory 0.9.3。训练需要支持 BF16 且显存足够的 GPU；下面的结果重算无需 GPU。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install 'torch==2.8.0' --index-url https://download.pytorch.org/whl/cu128
python -m pip install -c constraints.txt -r requirements.txt
llamafactory-cli version
```

历史依赖快照见 [environment_snapshot.txt](docs/environment_snapshot.txt)。快照来自当时虚拟环境及其继承的包，不等于最小依赖，也不建议整份安装。旧版环境可能需要单独排查平台兼容性。

### 2. 数据

本仓库不复制训练数据或模型权重。数据请从官方来源下载并遵守其条款。下载脚本列出 JSON 候选并核对历史 SHA256；多个文件时需显式传 `--filename`，源文件发生变化则拒绝继续。

```bash
python scripts/download_data.py
# 如有多个 JSON：python scripts/download_data.py --filename '<官方仓库中的JSON路径>'
python scripts/prepare_data.py
python scripts/prepare_label_only.py
```

若网络无法下载，可将已取得的原始 JSON 放到 `data/rank1_official_train.json`，先核对 [data_manifest.json](data/data_manifest.json) 中的 SHA256。运行 `prepare_data.py` 会重写生成的数据清单，比较历史版本前请保留副本。派生文件校验值见 [subset_checksums.json](data/subset_checksums.json)。

### 3. 基础模型

为了保持历史版本，请先下载已记录的模型提交：

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-1.5B', revision='8faed761d45a263340a0528343f099c05c9a4323', local_dir='models/Qwen2.5-1.5B')
PY
```

### 4. 两组分别从基础模型训练

```bash
llamafactory-cli train configs/qwen_1_5b_lora.yaml model_name_or_path=./models/Qwen2.5-1.5B
llamafactory-cli train configs/qwen_1_5b_label_only.yaml model_name_or_path=./models/Qwen2.5-1.5B
```

LoRA rank=32、alpha=64、目标为框架选中的线性层；**还完整训练 embed_tokens 和 lm_head**，不是纯 LoRA 小适配器。新增 `<think>` 和 `</think>` 后扩展词表。学习率 1e-4、一轮、单卡 batch=1、梯度累积=16、seed=12345、cutoff=2500。历史日志显示 312 次更新、epoch 0.9984，不声称每条训练样本均被完整处理。

### 5. 逐样本生成评估

```bash
python scripts/evaluate.py base --model ./models/Qwen2.5-1.5B
python scripts/evaluate.py finetuned --model ./models/Qwen2.5-1.5B
python scripts/evaluate.py label_only --model ./models/Qwen2.5-1.5B
```

本脚本默认离线加载，须先取得本地基础模型。输出到 `runs/`，遇到同名结果拒绝覆盖。规则沿用历史实现：greedy、最多 768 个新 token；输出完整标签即停止，接受直接的 true/false 或 `</think>` 后的标签。它不是官方连续概率评分器。停止规则不会代替模型选标签，但也不意味着模型学会了 EOS。

### 6. 无 GPU 核对既有结果

```bash
python scripts/summarize.py --output runs/summary_check.json
```

只需 Python 标准库即可校验 3×500 条预测、重新统计指标和配对计数。有 `data/val_500.json` 时额外按查询 bootstrap；没有时仍可验证准确率，但不会补造区间。要复现已提交的区间：先按历史数据哈希重建验证集，再运行同一命令。

## 目录与来源

```text
configs/                  可迁移的训练配置
scripts/                  数据准备、评估与结果核对
results/predictions/      三组真实逐样本预测及历史汇总
data/                     数据来源、划分、子集校验值（无训练原文）
docs/                     环境记录、证据来源及限制
archive/server_scripts/   恢复的历史服务器脚本，保留硬编码路径
archive/server_configs/   恢复的历史配置
```

`archive/` 是历史证据，默认不直接运行；顶层脚本只调整路径、命令行接口及输出位置，并保留历史生成与解析规则。`summarize.py` 和下载/仅标签准备脚本为整理仓库时新增的辅助工具，不声称它们是最初训练时运行的原文件。项目在 AI 编程助手辅助下完成代码与实验整理；方法和教师数据来自原作者。

## 结论

在当前小模型、少量训练数据、同来源分类验证设置下，两种微调方案均优于未微调基线，未观察到推理文本监督的额外准确率优势。若进一步检验原论文结论，需要独立评估集、多个训练种子，以及 BRIGHT 上相同候选集合的概率评分与 nDCG@10。
