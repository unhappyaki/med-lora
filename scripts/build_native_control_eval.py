from __future__ import annotations

import argparse
import csv
import html
import json
import re
from collections import OrderedDict
from pathlib import Path


def strip_html(value: str) -> str:
    value = html.unescape(value).replace("\u00a0", " ")
    value = re.sub(
        r"<\s*(?:br|/p|/div|/tr|/li|/h[1-6])\b[^>]*>",
        "\n",
        value,
        flags=re.I,
    )
    value = re.sub(r"<[^>]*>", "", value)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[ \t]+\n", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def load_prompt_template(path: Path) -> tuple[str, str]:
    with path.open("r", encoding="utf-8") as handle:
        sample = json.loads(next(line for line in handle if line.strip()))
    system = sample["messages"][0]["content"]
    user = sample["messages"][1]["content"]
    if "Q: " not in user:
        raise ValueError("训练样本不包含 Q: 标记")
    return system, user.split("Q: ", 1)[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--prompt-source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with Path(args.csv).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    system, prefix = load_prompt_template(Path(args.prompt_source))

    groups: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
    for row in rows:
        groups.setdefault(row["case"], []).append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for case, case_rows in groups.items():
            first = case_rows[0]
            raw_sources = {row["病历全文"] for row in case_rows}
            if len(raw_sources) != 1:
                raise ValueError(f"case={case} 存在多个病历全文")
            source = strip_html(next(iter(raw_sources)))
            gold = {
                "extractions": [
                    {
                        "slot": row["gold_清洗后"].strip(),
                        "slot_attributes": {"field_name": row["控件名"].strip()},
                    }
                    for row in case_rows
                ]
            }
            item = {
                "id": f"sharingcode::{first['医院']}::{case}",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"{prefix}Q: {source}\nA:"},
                    {
                        "role": "assistant",
                        "content": json.dumps(gold, ensure_ascii=False),
                    },
                ],
                "metadata": {
                    "hospital": first["医院"],
                    "document_type": first["分类名"],
                    "case": case,
                    "target_controls": len(case_rows),
                    "hard_case": False,
                },
            }
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(json.dumps({"cases": len(groups), "rows": len(rows), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
