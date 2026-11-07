# Qwen3.5-0.8B 病历抽取 v5 LoRA 实验

## 1. 本次实验固定口径

- 模型：`Qwen/Qwen3.5-0.8B`
- 提示词：全部使用 Gold 评测表中的唯一 `提示词_v5`
- 一条样本：一份完整病历，一次输出该病例的全部槽位
- 输出格式：LangExtract 实际 wire format，即 `extractions -> slot / slot_attributes.field_name`
- 数据过滤：排除 `section` 以“段落”开头、section 为空、Gold 为空的记录
- 训练目标：所有 `slot` 都必须是病历原文中的连续子串
- 划分方式：按病例划分训练集和内部验证集，同一病例不会跨集合

内部验证集只用于本轮选参数和判断微调是否有效。以后建立正式测试集时，不能从本次训练集或内部验证集复用病例。

## 2. 服务器目录建议

```text
/root/qwen35-0.8b-base/           原始模型
/root/medical-v5/data/            v5 数据
/root/medical-v5/scripts/         训练与评测脚本
/root/medical-v5/runs/            LoRA、日志和评测结果
```

现有 `/root/qwen35-deploy/model` 不参与本实验，也不删除。

## 3. 安装依赖

服务器已有 PyTorch、CUDA 和 Transformers，仅补充 LoRA 依赖：

```bash
python -m pip install -r /root/medical-v5/scripts/requirements-v5.txt
```

下载 0.8B 模型：

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-0.8B', local_dir='/root/qwen35-0.8b-base')"
```

## 4. 先看真实 token 长度

```bash
python /root/medical-v5/scripts/analyze_v5_lengths.py \
  --model /root/qwen35-0.8b-base \
  --data /root/medical-v5/data/train_v5_weighted.jsonl
```

训练脚本不静默截断病历。若最长样本超过 `16384`，根据统计提高 `--max-length`，或者单独复核异常超长病例。不要直接截掉病历尾部，否则尾部槽位会变成错误监督。

## 5. 跑未微调基线

```bash
python /root/medical-v5/scripts/evaluate_v5.py \
  --model /root/qwen35-0.8b-base \
  --data /root/medical-v5/data/validation_v5.jsonl \
  --output-dir /root/medical-v5/runs/baseline-v5
```

评测会生成：

- `predictions.jsonl`：逐病例原始输出，可中断续跑
- `metrics.json`：JSON 合法率、原文 grounding、字段精确率/召回率/F1、长段落召回率和文书类型分项

## 6. LoRA 训练

第一轮建议：

```bash
python /root/medical-v5/scripts/train_lora_v5.py \
  --model /root/qwen35-0.8b-base \
  --train-data /root/medical-v5/data/train_v5_weighted.jsonl \
  --validation-data /root/medical-v5/data/validation_v5.jsonl \
  --output-dir /root/medical-v5/runs/lora-r16-v5 \
  --epochs 3 \
  --max-length 16384 \
  --gradient-accumulation 8 \
  --learning-rate 2e-4 \
  --lora-rank 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05
```

脚本使用 bf16、batch size 1、梯度检查点和动态单样本长度，只对文本语言模型中的投影层注入 LoRA，自动排除视觉模块。

训练过程中重点看：

- `loss` 是否持续下降但没有突然变成 `nan`
- `max_memory_gb` 是否低于显卡上限
- 每轮的 `validation_loss` 是否继续改善
- `checkpoint-epoch-1/2/3` 都会保留，不能只按最后一轮判断

## 7. 同口径复评每个 checkpoint

```bash
python /root/medical-v5/scripts/evaluate_v5.py \
  --model /root/qwen35-0.8b-base \
  --adapter /root/medical-v5/runs/lora-r16-v5/checkpoint-epoch-1 \
  --data /root/medical-v5/data/validation_v5.jsonl \
  --output-dir /root/medical-v5/runs/eval-epoch-1
```

将 adapter 和输出目录分别替换为 epoch 2、epoch 3。选择模型时优先比较：

1. 严格字段 F1 是否明显超过原模型基线。
2. JSON 合法率和原文 grounding 率是否稳定。
3. 长文本槽位召回率是否提升。
4. 入院记录、首次病程记录、手术知情同意书的分项是否改善。
5. 是否出现过拟合：训练损失继续下降，但验证 F1 下降或误抽增加。

## 8. 如何判断 0.8B 微调可行

这轮是可行性验证，不以训练损失下降作为成功标准。满足以下条件才建议继续扩数据：

- 内部验证严格 F1 相比基线有明确提升。
- JSON 合法率接近 100%。
- 预测值大部分能在原文中找到，模型没有用常识改写病历。
- 困难文书的漏抽明显减少，同时误抽没有大幅增加。
- 换到未来独立测试集后仍保持提升。

如果主要问题仍是超长病历漏尾部、复杂长段落边界不完整，可继续补针对性病例；如果字段语义本身高度含糊，优先修订 Gold 和提示词定义，而不是盲目增加训练轮数。
