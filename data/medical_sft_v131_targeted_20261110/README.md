# V13.1 Targeted Boundary Dataset

此目录只公开数据统计与完整性指纹。含真实病历正文的文件保存在私有仓库 `unhappyaki/med-lora-data`：

- `v131_targeted_train.jsonl`：254 条
- `v131_targeted_validation.jsonl`：48 条
- `v131_real_holdout_relabelled.jsonl`：50 条固定真实留出集

数据质量审计：三个 split 的源文本交叉为 0；HTML、病历正文 `{}` 残留和超过 4K 的源病历均为 0；JSON、Schema、字段唯一性和严格 grounding 审计通过。

训练或推理容器内建议克隆私有数据仓到 `/root/medical-v5/private-data`，并建立：

```bash
ln -sfn /root/medical-v5/private-data/datasets/medical_sft_v131_targeted_20261110 \
  /root/medical-v5/data/current
```

传输后使用 `private_files.sha256` 对照校验。公开仓库不应加入这三个 JSONL 文件。
