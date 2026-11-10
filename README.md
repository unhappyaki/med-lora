# Medical LoRA

面向医疗病历信息抽取任务的 Qwen3.5-0.8B LoRA 微调实验。仓库包含数据准备、训练、评测脚本及对应的配置、适配器和离线评测结果。

> **数据与隐私：**本仓库是公开仓库。data/ 和 runs/ 中可能含有医疗文本、标注或模型产物。使用或再分发前，请确认数据授权、脱敏及公开许可；不要提交患者身份信息、访问令牌、密码或其他凭据。

## 项目结构

- scripts/：训练、数据准备、质量检查、评测和看板脚本。
- data/：不同版本的训练、验证、评测数据及审核结果。
- runs/：训练配置、适配器/checkpoint、训练记录和评测输出。
- dashboard/：训练看板状态文件。

含真实病历正文的新数据版本不再直接提交到本公开仓库。V13.1 原始 train、validation 和固定 holdout 保存在私有仓库 `unhappyaki/med-lora-data`；本仓库仅保留不含样本正文的统计、校验哈希、构建脚本与评测代码。

## V13.1 大 Section 边界抽取

| 数据 / 模型 | 结果 |
|---|---:|
| 训练 / 验证 / 固定真实 holdout | 254 / 48 / 50 |
| train / validation / holdout 源文本交叉 | 0 / 0 / 0 |
| V13.1 validation loss | 0.0002916 |
| 50 条 holdout 字段 P / R / F1（修订 Gold 前） | 0.9470 / 0.9434 / 0.9452 |
| 整例正确 | 36 / 50 |
| JSON / Schema 严格合法率 | 98% |

当前策略是先完成端到端评测和统一 Gold 修订，暂不继续追加训练。复盘见 `docs/V13.1训练评测与残留问题复盘_20261110.md`。

### V13.1 推理工作台

`scripts/v131_inference_workbench.py` 提供带鉴权的浏览器工作台、`/extract` 抽取接口和 OpenAI 兼容接口；`scripts/start_v131_workbench.sh` 将模型常驻在 GPU 并默认监听 `0.0.0.0:9876`。工作台会执行病历 `{}` 清洗、V2 大 Section Prompt、JSON 解析、Schema 校验和严格 grounding，并返回 token 数、耗时及生成速度。

当前生产测试后端使用常驻 Transformers，以稳定性优先。vLLM 已识别 Qwen3.5 与 LoRA，但首次运行需要完整的 FlashInfer 采样内核环境，完成兼容性验证前不替换稳定后端。

## 当前离线性能

单控件任务按字段评分；字段完全匹配不等于整份病历完全正确。不同评测集来源和样本不同，不能直接横向比较。

### 单控件抽取

| 评测集 | 模型 / checkpoint | 字段 / 病例 | 字段完全匹配 | 字符级 P / R / F1 | 预测原文命中 |
|---|---|---:|---:|---:|---:|
| V8 单控件验证集 | V8 single-control，epoch 1 | 92 / 19 | 75.0% (69/92) | 36.7% / 80.9% / 50.4% | 100% |
| 同上 | V8 single-control，epoch 2 | 92 / 19 | 70.7% (65/92) | 33.2% / 80.9% / 47.1% | 100% |
| V8 清理后验证子集 | V7 single-control r3，epoch 1 | 86 / 18 | 80.2% (69/86) | 36.3% / 98.6% / 53.0% | 100% |
| 同上 | V8 balanced-clean，epoch 1 | 86 / 18 | 77.9% (67/86) | 34.9% / 98.9% / 51.6% | 100% |
| 四院单控件评测集 | V7 single-control r3，epoch 1 | 443 / 33 | 15.3% (68/443) | 42.0% / 73.0% / 53.4% | 94.6% |
| 同上 | V7 single-control r3，epoch 2 | 443 / 33 | 14.9% (66/443) | 41.8% / 75.4% / 53.8% | 95.7% |

评测文件：V8 单控件集为 data/v8_20261106/validation_v8_single_control.csv；清理后子集为 data/v8_clean_validation_20261106_r5/validation_v8_single_control.csv；四院集为 data/eval_set_4hospital_sharingcode_ctrl.csv。

V8 单控件验证集上，epoch 1 的字段完全匹配率和字符级 F1 高于 epoch 2。四院集上，V7 两个 epoch 的字段完全匹配率约 15%，字符级 F1 约 53%。字符级分数高于完全匹配率，表示预测常与答案部分重合，但整段字段完全一致的比例仍低。

### 完整病历结构化抽取（较早的 V5/V6 验证）

| 评测集 | 模型 / checkpoint | 病例数 | JSON / schema 有效率 | 字段 P / R / F1 | 整病例全对 | 原文命中 |
|---|---|---:|---:|---:|---:|---:|
| validation_v5.jsonl | 基础模型，无 LoRA | 30 | 73.3% / 73.3% | 0.0% / 0.0% / 0.0% | 0% | 44.2% |
| 同上 | V5 main，epoch 3 | 30 | 96.7% / 93.3% | 87.4% / 81.5% / 84.3% | 36.7% | 99.7% |
| 同上 | V6 supplement，epoch 2 | 30 | 90.0% / 90.0% | 90.5% / 87.9% / 89.2% | 40.0% | 99.0% |

此表是不同任务和评测口径，样本为 30 例，应与单控件结果分开解读。结果文件：runs/evaluations/baseline-v5/metrics.json、runs/evaluations/eval-lora-r16-v5-main-01--checkpoint-epoch-3/metrics.json、runs/evaluations/v6-epoch-2-validation/metrics.json。

## 指标说明

### 单控件评测

- **字段完全匹配率 (exact accuracy)：**预测字段与标准答案逐字相同的字段数 / 字段总数。代码也算去除空白后的 normalized accuracy；表中列严格原文匹配率。它不是病例级准确率。
- **字符级精确率 (P)：**预测文本中匹配到标准答案的字符数 / 预测文本字符数。较低可能表示输出含有多余内容。
- **字符级召回率 (R)：**匹配字符数 / 标准答案字符数。较低可能表示答案内容有遗漏。
- **字符级 F1：**P 和 R 的调和平均数。评测先去除空白，再用匹配字符块计算重合度；部分重合也会得分，因此不同于完全匹配率。
- **预测原文命中率：**预测字段去除空白后，能在病历原文去除空白后的文本中找到的比例。命中只表示可定位，不证明内容完整、字段归属正确或临床判断正确。
- **字段 / 病例数：**字段是被评分的控件数；病例是不同病历数，一个病例可贡献多个字段。

### 完整病历评测

- **JSON 有效率：**输出可解析为 JSON 的病例比例。
- **Schema 有效率：**JSON 符合预期结构和字段约束的病例比例。
- **字段 P / R / F1：**按字段真阳性、误报、漏报计算，与单控件的字符级指标不是同一算法。
- **整病例全对率 (case exact rate)：**一份病例中的所有抽取结果均与标准答案一致的病例比例，要求高于单字段匹配。
- **原文命中率 (grounding rate)：**抽取内容能在病历原文中找到依据的比例；不衡量答案完整性或字段归属。
- **长段落召回率：**长文本抽取目标中成功覆盖内容的比例。

## 环境与脚本

依赖清单：scripts/requirements-v5.txt。

- 训练：scripts/train_lora_v5.py
- 完整病历评测：scripts/evaluate_v5.py
- 单控件评测：scripts/evaluate_single_control.py
- 数据准备与分析：scripts/prepare_*.py、scripts/analyze_v5_lengths.py

所需 PyTorch、Transformers、PEFT 版本和 GPU 条件请以实际环境及 requirements 文件为准。模型、训练参数和数据路径可参考 runs/ 对应运行目录的 run_config.json。运行前请核对脚本参数及基础模型许可。

## 评测限制

这些是离线实验结果，不代表正式部署表现、临床有效性或安全性。验证集规模有限，数据切分、医院来源和字段分布都会影响得分。实际使用前应进行独立数据验证、人工审核和适用的隐私及合规审查。
