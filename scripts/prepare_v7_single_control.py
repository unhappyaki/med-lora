from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import html
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import oracledb

from prepare_v6_db_supplement import (
    IDENTITY_FIELDS,
    SYSTEM_FIELD_NAMES,
    TEMPLATE_PLACEHOLDER_RE,
    choose_content,
    clean_rich_text,
    general_deidentify,
    read_lob,
    remove_source_boilerplate,
    sanitize_labeled_identities,
    stable_hash,
)


HOSPITAL_CODES = {
    "石狮市妇幼保健院": "shishi",
    "柳城县人民医院": "liucheng",
    "邵武市中医院": "shaowu",
    "泉州市儿童医院": "quanzhou_child",
}

GROUP_TARGETS = {
    "admission": 110,
    "first_progress": 65,
    "discharge": 60,
    "outpatient": 60,
    "surgery": 38,
}

GROUP_ORDER = tuple(GROUP_TARGETS)
GROUP_QUERY_LIMITS = {
    "admission": 500,
    "first_progress": 400,
    "discharge": 400,
    "surgery": 400,
}

# The current development databases contain only two recent ``手术记录``
# rows and both have empty rich-text bodies.  V6 already contains ten
# independently audited, source-traceable operation notes from the same four
# hospitals.  Convert at most three clinically useful fields from each note
# into the V7 one-control task, and remove the corresponding V6 multi-field
# replay row so the same source is not overweighted twice.
SURGERY_FIELD_CODES = {
    "手术经过": "operation_process",
    "手术名称": "operation_name",
    "术前诊断": "preoperative_diagnosis",
    "术中诊断": "intraoperative_diagnosis",
    "术后诊断": "postoperative_diagnosis",
    "病理诊断": "pathological_diagnosis",
    "麻醉方式": "anesthesia_method",
    "出血量": "blood_loss",
    "标本": "specimen",
}
SURGERY_FIELD_PRIORITY = tuple(SURGERY_FIELD_CODES)

IDENTITY_LIKE_NAMES = set(IDENTITY_FIELDS) | {
    "姓名",
    "患者姓名",
    "患儿姓名",
    "身份证号",
    "手机号",
    "联系人",
    "联系人姓名",
    "联系人电话",
    "家庭住址",
    "详细地址",
    "医生签名",
    "护士签名",
}

PLACEHOLDER_RE = re.compile(
    r"(?:\{[^{}]{1,80}\}|\$\{[^{}]{1,80}\}|<%[^%]{1,80}%>|"
    r"(?:请|待)(?:填写|录入|补充|选择)|请输入|点击(?:此处|选择)|"
    r"(?:控件|字段)(?:代码|名称)?[:：]?\s*[A-Za-z_][A-Za-z0-9_.-]*)",
    re.I,
)

GENERIC_TEMPLATE_RE = re.compile(
    r"^(?:无|正常|未见异常|阴性|阳性|否|是|不详|未提供|暂无|暂缺)$"
)

CONTROL_CODE_RE = re.compile(r"^(?:DE\d{2}(?:\.\d{2}){3}|[A-Za-z][A-Za-z0-9_.-]{2,80})$")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def compact_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "")
    value = html.unescape(value).replace("\u00a0", " ")
    return re.sub(r"\s+", "", value)


def source_fingerprint(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "")
    value = re.sub(r"(?<!\d)1[3-9]\d{9}(?!\d)", "PHONE", value)
    value = re.sub(r"(?<![0-9A-Za-z])\d{17}[0-9Xx](?![0-9A-Za-z])", "IDCARD", value)
    value = re.sub(r"\d+", "#", value)
    return re.sub(r"[^A-Za-z#\u4e00-\u9fff]", "", value).lower()


def ngram_signature(value: str, size: int = 12, step: int = 4) -> set[str]:
    if len(value) < size:
        return {value} if value else set()
    return {value[index : index + size] for index in range(0, len(value) - size + 1, step)}


def near_duplicate(candidate: str, forbidden: list[tuple[str, set[str], str]]) -> tuple[bool, str]:
    fingerprint = source_fingerprint(candidate)
    if not fingerprint:
        return False, ""
    signature = ngram_signature(fingerprint)
    for other, other_signature, label in forbidden:
        ratio = min(len(fingerprint), len(other)) / max(len(fingerprint), len(other), 1)
        if ratio < 0.78:
            continue
        if fingerprint == other:
            return True, label
        if not signature or not other_signature:
            continue
        union = len(signature | other_signature)
        jaccard = len(signature & other_signature) / union if union else 0.0
        if jaccard >= 0.82:
            return True, label
        if jaccard >= 0.68 and ratio >= 0.92:
            similarity = difflib.SequenceMatcher(None, fingerprint, other, autojunk=False).ratio()
            if similarity >= 0.94:
                return True, label
    return False, ""


def source_from_sample(sample: dict) -> str:
    messages = sample.get("messages") or []
    user = next((str(item.get("content") or "") for item in messages if item.get("role") == "user"), "")
    if "【病历全文】" in user and "【需抽取的控件】" in user:
        return user.split("【病历全文】", 1)[1].split("【需抽取的控件】", 1)[0].strip()
    if "Q: " in user:
        return user.split("Q: ", 1)[1].rsplit("\nA:", 1)[0].strip()
    return user


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_eval_context(eval_csv: Path, analysis_json: Path) -> tuple[str, list[str], list[tuple[str, set[str], str]], dict[tuple[str, str], float], dict[str, float]]:
    with eval_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("四院验证集为空")
    system_prompts = [row.get("真实提示词_system", "").strip() for row in rows if row.get("真实提示词_system", "").strip()]
    system_prompt = Counter(system_prompts).most_common(1)[0][0]
    cases = sorted({row.get("case", "").strip() for row in rows if row.get("case", "").strip()})
    unique_sources = sorted({clean_rich_text(row.get("病历全文", "")) for row in rows if row.get("病历全文", "").strip()})
    forbidden = []
    for source in unique_sources:
        fp = source_fingerprint(source)
        forbidden.append((fp, ngram_signature(fp), "four_hospital_eval"))

    by_pair: dict[tuple[str, str], float] = {}
    by_name: dict[str, float] = {}
    if analysis_json.exists():
        analysis = json.loads(analysis_json.read_text(encoding="utf-8"))
        for row in analysis.get("by_control", []):
            name = str(row.get("控件名") or "").strip()
            code = str(row.get("控件代码") or "").strip()
            count = float(row.get("样本数") or 0)
            exact_rate = float(row.get("精确率") or 0)
            f1 = float(row.get("字符F1") or 0)
            score = count * (1.0 - exact_rate) + 3.0 * (1.0 - f1)
            if name or code:
                by_pair[(name, code)] = max(score, by_pair.get((name, code), 0.0))
            if name:
                by_name[name] = max(score, by_name.get(name, 0.0))
    return system_prompt, cases, forbidden, by_pair, by_name


def add_forbidden_from_jsonl(forbidden: list[tuple[str, set[str], str]], path: Path, label: str) -> None:
    for sample in load_jsonl(path):
        source = source_from_sample(sample)
        if not source:
            continue
        fp = source_fingerprint(source)
        forbidden.append((fp, ngram_signature(fp), label))


def table_columns(cursor, table_name: str) -> set[str]:
    cursor.execute("select column_name from user_tab_columns where table_name=:name", name=table_name.upper())
    return {str(row[0]).upper() for row in cursor}


def query_inpatient(cursor, group: str, limit: int) -> list[dict]:
    conditions = {
        "admission": "(f.mingcheng like '%入院记录%' or f.mingcheng like '%入院病历%')",
        "first_progress": "f.mingcheng like '%首次病程记录%'",
        "discharge": "(f.mingcheng like '%出院记录%' or f.mingcheng like '%出院小结%')",
        "surgery": "f.mingcheng like '%手术记录%'",
    }
    cursor.execute(
        f"""
        select * from (
          select b.bingliid, b.xingming, b.jzys, b.zzys, b.zrys, b.sxys,
                 b.fenleiid, f.mingcheng,
                 mx.jzneirong, mx.zzneirong, mx.zrneirong, mx.gdneirong, mx.sxneirong
          from emr_z_bingli b
          join emr_z_binglimx mx on mx.bingliid=b.bingliid
          join emr_z_blfenlei f on f.fenleiid=b.fenleiid
          where {conditions[group]}
          order by b.blshijian desc nulls last
        ) where rownum <= {int(limit)}
        """
    )
    output = []
    for row in cursor:
        output.append({
            "record_id": str(row[0]),
            "patient_name": str(row[1] or ""),
            "doctor_names": [str(value or "") for value in row[2:6]],
            "category_id": str(row[6]),
            "document_type": str(row[7]),
            "content": choose_content(list(row[8:13])),
            "setting": "inpatient",
            "group": group,
        })
    return output


def query_outpatient(cursor, limit: int = 700) -> list[dict]:
    columns = table_columns(cursor, "EMR_M_LCBINGLIMX")
    xml_expression = "mx.xmlneirong" if "XMLNEIRONG" in columns else "null"
    cursor.execute(
        f"""
        select * from (
          select b.bingliid, b.xingming, b.yonghuming, b.fenleiid, f.mingcheng,
                 mx.neirong, {xml_expression}
          from emr_m_lcbingli b
          join emr_m_lcbinglimx mx on mx.bingliid=b.bingliid
          join emr_z_blfenlei f on f.fenleiid=b.fenleiid
          where f.mingcheng in ('初诊','复诊','急诊')
          order by b.shijian desc nulls last
        ) where rownum <= {int(limit)}
        """
    )
    output = []
    for row in cursor:
        output.append({
            "record_id": str(row[0]),
            "patient_name": str(row[1] or ""),
            "doctor_names": [str(row[2] or "")],
            "category_id": str(row[3]),
            "document_type": str(row[4]),
            "content": choose_content(list(row[5:7])),
            "setting": "outpatient",
            "group": "outpatient",
        })
    return output


def load_controls(cursor, record_id: str, category_id: str) -> list[dict]:
    r_columns = table_columns(cursor, "EMR_Z_RECORDFORSHARING")
    x_columns = table_columns(cursor, "EMR_X_SHARINGCODE")
    r_name = "r.mingcheng" if "MINGCHENG" in r_columns else "null"
    x_name = "x.mingcheng" if "MINGCHENG" in x_columns else "null"
    r_field = "r.field" if "FIELD" in r_columns else "null"
    x_field = "x.field" if "FIELD" in x_columns else "null"
    r_code = "r.de_code" if "DE_CODE" in r_columns else "null"
    x_code = "x.de_code" if "DE_CODE" in x_columns else "null"
    join_parts = []
    if "DE_CODE" in r_columns and "DE_CODE" in x_columns:
        join_parts.append("x.de_code=r.de_code")
    if "FIELD" in r_columns and "FIELD" in x_columns:
        join_parts.append("(x.field=r.field and x.field is not null)")
    join_condition = " or ".join(join_parts) or "1=0"
    cursor.execute(
        f"""
        select nvl({r_name}, nvl({x_name}, nvl({r_field}, nvl({x_field}, {r_code})))) control_name,
               nvl({r_code}, {x_code}) control_code,
               nvl({r_field}, {x_field}) control_field,
               r.value
        from emr_z_recordforsharing r
        left join emr_x_sharingcode x
          on x.fenleiid=:category_id and ({join_condition})
        where r.bingliid=:record_id
          and r.value is not null
          and length(trim(r.value)) > 0
        """,
        category_id=category_id,
        record_id=record_id,
    )
    output = []
    seen = set()
    for name, code, field, value in cursor:
        item = {
            "control_name": re.sub(r"\s+", "", str(name or "")).strip("：:"),
            "control_code": str(code or field or "").strip(),
            "control_field": str(field or "").strip(),
            "raw_value": read_lob(value).strip(),
        }
        key = (item["control_name"], item["control_code"], item["raw_value"])
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def strip_control_prefix(value: str, control_name: str, source: str) -> tuple[str, bool]:
    pattern = re.compile(rf"^\s*{re.escape(control_name)}\s*[：:]\s*", re.I)
    stripped = pattern.sub("", value, count=1).strip()
    if stripped and stripped != value and stripped in source:
        return stripped, True
    return value, False


def build_user_prompt(source: str, control_name: str, control_code: str) -> str:
    return (
        f"【病历全文】\n{source}\n\n"
        f"【需抽取的控件】\n名称：{control_name}\n代码：{control_code}"
    )


def is_label_only_value(value: str, control_name: str, control_code: str = "") -> bool:
    compact_value = compact_text(value)
    compact_name = compact_text(control_name)
    label_without_terminal = re.sub(r"[：:]+$", "", compact_value).strip()
    return (
        value == control_name
        or bool(control_code and value == control_code)
        or compact_value == compact_name
        or label_without_terminal == compact_name
        or bool(re.fullmatch(r"[^\r\n：:]{1,40}[：:]", value.strip()))
    )


def evaluate_control(record: dict, source: str, control: dict, hospital: str, priority_pair: dict[tuple[str, str], float], priority_name: dict[str, float]) -> tuple[dict | None, str, list[str]]:
    name = control["control_name"].strip()
    code = control["control_code"].strip()
    raw_value = control["raw_value"]
    reasons: list[str] = []
    if not name or name.startswith("段落") or name in SYSTEM_FIELD_NAMES:
        return None, "excluded", ["paragraph_or_system_field"]
    if name in IDENTITY_LIKE_NAMES or any(token in name for token in ("姓名", "住址", "电话", "身份证", "签名")):
        return None, "excluded", ["identity_field"]
    if not code or not CONTROL_CODE_RE.match(code):
        return None, "excluded", ["invalid_control_code"]
    value = clean_rich_text(raw_value)
    value = general_deidentify(value, record["patient_name"], record["doctor_names"]).strip()
    value, prefix_removed = strip_control_prefix(value, name, source)
    visible_length = len(re.sub(r"\s+", "", value))
    if not value or visible_length < 2:
        return None, "excluded", ["value_too_short"]
    if visible_length > 1200:
        return None, "excluded", ["value_too_long"]
    # A sharing-control row can contain only the rendered label when the
    # underlying field is blank (for example, ``现病史:``).  Mere source
    # traceability would incorrectly accept that label as the gold value.
    # Reject both exact label echoes and any short, label-shaped value that
    # ends at a colon without containing an actual value.
    if is_label_only_value(value, name, code):
        return None, "excluded", ["label_or_code_echo"]
    if TEMPLATE_PLACEHOLDER_RE.search(value) or PLACEHOLDER_RE.search(value):
        return None, "excluded", ["template_placeholder"]
    if re.search(r"<\s*/?\s*[A-Za-z_][^>]*>", value):
        return None, "excluded", ["html_residual"]
    if value not in source:
        return None, "excluded", ["not_traceable_to_source"]
    occurrences = source.count(value)
    if len(source) > 16000:
        return None, "excluded", ["source_too_long"]
    if len(source) < 100:
        return None, "excluded", ["source_too_short"]
    if GENERIC_TEMPLATE_RE.match(value) and occurrences > 5:
        reasons.append("generic_value_many_occurrences")
    if occurrences > 8 and visible_length <= 6:
        reasons.append("ambiguous_repeated_short_value")
    if visible_length > 800:
        reasons.append("very_long_target")
    if len(source) > 12000:
        reasons.append("very_long_source")
    status = "flagged" if reasons else "auto_pass"
    priority = priority_pair.get((name, code), priority_name.get(name, 0.0))
    public_case = (
        f"v7db-{HOSPITAL_CODES[hospital]}-{record['group']}-"
        f"{stable_hash(hospital + '|' + record['setting'] + '|' + record['record_id'])[:16]}"
    )
    sample_id = f"{public_case}::{stable_hash(name + '|' + code + '|' + value)[:12]}"
    item = {
        "id": sample_id,
        "messages": [],
        "metadata": {
            "hospital": hospital,
            "document_type": record["document_type"],
            "case": public_case,
            "control_name": name,
            "control_code": code,
            "control_field": control["control_field"],
            "source_type": "single_control",
            "source_setting": record["setting"],
            "target_group": record["group"],
            "source_chars": len(source),
            "target_chars": len(value),
            "source_occurrences": occurrences,
            "priority_score": round(priority, 6),
            "priority_from_v6_errors": priority > 0,
            "prefix_removed": prefix_removed,
            "review_status": status,
            "review_reasons": reasons,
            "html_removed": True,
            "deidentified": True,
            "strict_source_traceable": True,
        },
        "_source": source,
        "_gold": value,
    }
    return item, status, reasons


def deterministic_key(item: dict) -> tuple:
    metadata = item["metadata"]
    return (
        -float(metadata.get("priority_score") or 0),
        stable_hash(item["id"]),
    )


def balanced_select(items: list[dict], targets: dict[str, int], total_target: int) -> list[dict]:
    by_group_hospital: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in items:
        metadata = item["metadata"]
        by_group_hospital[(metadata["target_group"], metadata["hospital"])].append(item)
    for bucket in by_group_hospital.values():
        bucket.sort(key=deterministic_key)

    selected: list[dict] = []
    used_ids: set[str] = set()
    per_case = Counter()
    per_control = Counter()
    per_hospital = Counter()
    hospitals = sorted(HOSPITAL_CODES)

    def take(group: str, hospital: str, quota: int) -> int:
        taken = 0
        bucket = by_group_hospital.get((group, hospital), [])
        for item in bucket:
            if taken >= quota:
                break
            if item["id"] in used_ids:
                continue
            metadata = item["metadata"]
            case = metadata["case"]
            control_key = (metadata["control_name"], metadata["control_code"])
            if per_case[case] >= 3 or per_control[control_key] >= 14:
                continue
            selected.append(item)
            used_ids.add(item["id"])
            per_case[case] += 1
            per_control[control_key] += 1
            per_hospital[hospital] += 1
            taken += 1
        return taken

    for group in GROUP_ORDER:
        group_target = min(targets.get(group, 0), total_target - len(selected))
        if group_target <= 0:
            break
        base = group_target // len(hospitals)
        remainder = group_target % len(hospitals)
        for index, hospital in enumerate(hospitals):
            take(group, hospital, base + (1 if index < remainder else 0))
        missing = group_target - sum(1 for item in selected if item["metadata"]["target_group"] == group)
        if missing > 0:
            merged = sorted(
                [item for hospital in hospitals for item in by_group_hospital.get((group, hospital), [])],
                key=deterministic_key,
            )
            for item in merged:
                if missing <= 0:
                    break
                if item["id"] in used_ids:
                    continue
                metadata = item["metadata"]
                case = metadata["case"]
                control_key = (metadata["control_name"], metadata["control_code"])
                if per_case[case] >= 3 or per_control[control_key] >= 14:
                    continue
                selected.append(item)
                used_ids.add(item["id"])
                per_case[case] += 1
                per_control[control_key] += 1
                per_hospital[metadata["hospital"]] += 1
                missing -= 1

    if len(selected) < total_target:
        remaining = sorted((item for item in items if item["id"] not in used_ids), key=deterministic_key)
        for item in remaining:
            if len(selected) >= total_target:
                break
            metadata = item["metadata"]
            case = metadata["case"]
            control_key = (metadata["control_name"], metadata["control_code"])
            if per_case[case] >= 3 or per_control[control_key] >= 14:
                continue
            selected.append(item)
            used_ids.add(item["id"])
            per_case[case] += 1
            per_control[control_key] += 1
    return sorted(selected, key=lambda item: item["id"])


def finalize_sample(item: dict, system_prompt: str) -> dict:
    source = item.pop("_source")
    gold = item.pop("_gold")
    metadata = item["metadata"]
    item["messages"] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": build_user_prompt(source, metadata["control_name"], metadata["control_code"]),
        },
        {"role": "assistant", "content": gold},
    ]
    return item


def prepare_v6_surgery_single_control(
    path: Path,
    forbidden: list[tuple[str, set[str], str]],
    priority_pair: dict[tuple[str, str], float],
    priority_name: dict[str, float],
) -> tuple[list[dict], list[dict], set[str]]:
    output: list[dict] = []
    audit: list[dict] = []
    converted_ids: set[str] = set()
    for sample in load_jsonl(path):
        metadata = sample.get("metadata") or {}
        if metadata.get("target_group") != "surgery":
            continue
        source = source_from_sample(sample)
        hospital = str(metadata.get("source_hospital") or "")
        source_id = str(sample.get("id") or "")
        duplicate, duplicate_reason = near_duplicate(source, forbidden)
        if duplicate or hospital not in HOSPITAL_CODES:
            audit.append({
                "hospital": hospital,
                "group": "surgery",
                "setting": metadata.get("source_setting", "inpatient"),
                "document_type": metadata.get("document_type", "手术记录"),
                "status": "excluded",
                "reason": duplicate_reason if duplicate else "unknown_hospital",
            })
            continue
        try:
            payload = json.loads(sample["messages"][-1]["content"])
            extractions = list(payload.get("extractions") or [])
        except (KeyError, TypeError, json.JSONDecodeError):
            audit.append({
                "hospital": hospital,
                "group": "surgery",
                "setting": metadata.get("source_setting", "inpatient"),
                "document_type": metadata.get("document_type", "手术记录"),
                "status": "excluded",
                "reason": "invalid_v6_gold_json",
            })
            continue

        by_name: dict[str, list[str]] = defaultdict(list)
        for extraction in extractions:
            name = str((extraction.get("slot_attributes") or {}).get("field_name") or "").strip()
            value = str(extraction.get("slot") or "").strip()
            if name in SURGERY_FIELD_CODES and value:
                by_name[name].append(value)

        case = f"v7v6-surgery-{stable_hash(source_id)[:16]}"
        accepted_for_case = 0
        for name in SURGERY_FIELD_PRIORITY:
            values = by_name.get(name, [])
            # Repeated fields can represent separate fragments.  A single
            # answer would then be incomplete, so leave them for review
            # rather than teaching a partial value.
            if len(values) != 1:
                continue
            value = values[0]
            code = SURGERY_FIELD_CODES[name]
            reason = ""
            if len(re.sub(r"\s+", "", value)) < 2:
                reason = "value_too_short"
            elif is_label_only_value(value, name, code):
                reason = "label_or_code_echo"
            elif TEMPLATE_PLACEHOLDER_RE.search(value) or PLACEHOLDER_RE.search(value):
                reason = "template_placeholder"
            elif re.search(r"<\s*/?\s*[A-Za-z_][^>]*>", value):
                reason = "html_residual"
            elif value not in source:
                reason = "not_traceable_to_source"
            if reason:
                audit.append({
                    "hospital": hospital,
                    "group": "surgery",
                    "setting": metadata.get("source_setting", "inpatient"),
                    "document_type": metadata.get("document_type", "手术记录"),
                    "control_name": name,
                    "control_code": code,
                    "status": "excluded",
                    "reason": reason,
                    "source_chars": len(source),
                    "target_chars": len(value),
                    "gold": value,
                    "source_excerpt": source_excerpt(source, value),
                })
                continue
            priority = priority_pair.get((name, code), priority_name.get(name, 0.0))
            sample_id = f"{case}::{stable_hash(name + '|' + code + '|' + value)[:12]}"
            item = {
                "id": sample_id,
                "messages": [],
                "metadata": {
                    "hospital": hospital,
                    "document_type": metadata.get("document_type", "手术记录"),
                    "case": case,
                    "control_name": name,
                    "control_code": code,
                    "control_field": code,
                    "source_type": "single_control_v6_audited_surgery",
                    "source_setting": metadata.get("source_setting", "inpatient"),
                    "target_group": "surgery",
                    "source_chars": len(source),
                    "target_chars": len(value),
                    "source_occurrences": source.count(value),
                    "priority_score": round(priority, 6),
                    "priority_from_v6_errors": priority > 0,
                    "prefix_removed": False,
                    "review_status": "auto_pass",
                    "review_reasons": [],
                    "html_removed": True,
                    "deidentified": True,
                    "strict_source_traceable": True,
                },
                "_source": source,
                "_gold": value,
            }
            output.append(item)
            audit.append({
                "sample_id": sample_id,
                "hospital": hospital,
                "group": "surgery",
                "setting": metadata.get("source_setting", "inpatient"),
                "document_type": metadata.get("document_type", "手术记录"),
                "case": case,
                "control_name": name,
                "control_code": code,
                "status": "auto_pass",
                "reason": "converted_from_v6_audited_surgery",
                "source_chars": len(source),
                "target_chars": len(value),
                "source_occurrences": source.count(value),
                "priority_score": round(priority, 6),
                "gold": value,
                "source_excerpt": source_excerpt(source, value),
            })
            accepted_for_case += 1
            if accepted_for_case >= 3:
                break
        if accepted_for_case:
            converted_ids.add(source_id)
    return output, audit, converted_ids


def prepare_replay(
    paths: list[tuple[Path, str]],
    forbidden: list[tuple[str, set[str], str]],
    excluded_ids: set[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    output = []
    audit = []
    seen_ids = set()
    excluded_ids = excluded_ids or set()
    for path, source_type in paths:
        for sample in load_jsonl(path):
            source = source_from_sample(sample)
            is_forbidden, reason = near_duplicate(source, forbidden)
            row = {
                "id": sample.get("id", ""),
                "source_type": source_type,
                "status": "excluded" if is_forbidden or sample.get("id") in excluded_ids else "included",
                "reason": (
                    "converted_to_single_control"
                    if sample.get("id") in excluded_ids
                    else reason if is_forbidden else ""
                ),
            }
            if sample.get("id") in seen_ids:
                row["status"] = "excluded"
                row["reason"] = "duplicate_replay_id"
            audit.append(row)
            if row["status"] == "excluded":
                continue
            seen_ids.add(sample.get("id"))
            metadata = dict(sample.get("metadata") or {})
            metadata["source_type"] = source_type
            metadata["replay_preserves_original_task"] = True
            sample["metadata"] = metadata
            output.append(sample)
    return output, audit


def source_excerpt(source: str, gold: str, radius: int = 140) -> str:
    index = source.find(gold)
    if index < 0:
        return source[: radius * 2]
    start = max(0, index - radius)
    end = min(len(source), index + len(gold) + radius)
    excerpt = source[start:end]
    if start:
        excerpt = "…" + excerpt
    if end < len(source):
        excerpt += "…"
    return excerpt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-csv", required=True)
    parser.add_argument("--analysis-json", required=True)
    parser.add_argument("--v5-validation", required=True)
    parser.add_argument("--v5-replay", required=True)
    parser.add_argument("--v6-replay", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--new-target", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    system_prompt, eval_cases, forbidden, priority_pair, priority_name = load_eval_context(
        Path(args.eval_csv), Path(args.analysis_json)
    )
    add_forbidden_from_jsonl(forbidden, Path(args.v5_validation), "v5_validation")

    surgery_candidates, surgery_audit, converted_surgery_ids = prepare_v6_surgery_single_control(
        Path(args.v6_replay), forbidden, priority_pair, priority_name
    )

    replay, replay_audit = prepare_replay(
        [
            (Path(args.v5_replay), "replay_v5"),
            (Path(args.v6_replay), "replay_v6"),
        ],
        forbidden,
        converted_surgery_ids,
    )
    new_target = args.new_target or round(len(replay) * 1.5)
    scaled_targets = dict(GROUP_TARGETS)
    if new_target != sum(GROUP_TARGETS.values()):
        scale = new_target / sum(GROUP_TARGETS.values())
        scaled_targets = {group: round(value * scale) for group, value in GROUP_TARGETS.items()}
        difference = new_target - sum(scaled_targets.values())
        for group in GROUP_ORDER:
            if difference == 0:
                break
            scaled_targets[group] += 1 if difference > 0 else -1
            difference += -1 if difference > 0 else 1

    connections = json.loads(os.environ["V6_CONNECTIONS_JSON"])
    candidates: list[dict] = list(surgery_candidates)
    db_audit: list[dict] = list(surgery_audit)
    record_seen: set[tuple[str, str, str]] = set()
    control_seen: set[tuple[str, str, str, str]] = {
        (
            item["metadata"]["hospital"],
            item["metadata"]["target_group"],
            item["metadata"]["control_name"],
            compact_text(item["_gold"]),
        )
        for item in surgery_candidates
    }

    for hospital in sorted(connections):
        print(f"[hospital] {hospital}", flush=True)
        with oracledb.connect(**connections[hospital]) as database:
            with database.cursor() as cursor:
                records = []
                for group in ("admission", "first_progress", "discharge", "surgery"):
                    records.extend(query_inpatient(cursor, group, GROUP_QUERY_LIMITS[group]))
                records.extend(query_outpatient(cursor))
                records.sort(key=lambda row: stable_hash(hospital + "|" + row["group"] + "|" + row["record_id"]))
                for record in records:
                    record_key = (hospital, record["setting"], record["record_id"])
                    if record_key in record_seen:
                        continue
                    record_seen.add(record_key)
                    raw_source = record["content"]
                    if not raw_source:
                        db_audit.append({
                            "hospital": hospital,
                            "group": record["group"],
                            "document_type": record["document_type"],
                            "status": "excluded",
                            "reason": "empty_source",
                        })
                        continue
                    source = general_deidentify(raw_source, record["patient_name"], record["doctor_names"])
                    source = sanitize_labeled_identities(source)
                    source = remove_source_boilerplate(source).strip()
                    duplicate, duplicate_label = near_duplicate(source, forbidden)
                    if duplicate:
                        db_audit.append({
                            "hospital": hospital,
                            "group": record["group"],
                            "document_type": record["document_type"],
                            "status": "excluded",
                            "reason": duplicate_label,
                        })
                        continue
                    if re.search(r"<\s*/?\s*[A-Za-z_][^>]*>", source):
                        db_audit.append({
                            "hospital": hospital,
                            "group": record["group"],
                            "document_type": record["document_type"],
                            "status": "excluded",
                            "reason": "html_residual",
                        })
                        continue
                    controls = load_controls(cursor, record["record_id"], record["category_id"])
                    included_for_record = 0
                    for control in controls:
                        item, status, reasons = evaluate_control(
                            record, source, control, hospital, priority_pair, priority_name
                        )
                        audit_row = {
                            "sample_id": item["id"] if item else "",
                            "hospital": hospital,
                            "group": record["group"],
                            "setting": record["setting"],
                            "document_type": record["document_type"],
                            "case": item["metadata"]["case"] if item else "",
                            "control_name": control["control_name"],
                            "control_code": control["control_code"],
                            "status": status,
                            "reason": "|".join(reasons),
                            "source_chars": len(source),
                            "target_chars": len(item["_gold"]) if item else 0,
                            "source_occurrences": item["metadata"]["source_occurrences"] if item else 0,
                            "priority_score": item["metadata"]["priority_score"] if item else 0,
                            "gold": item["_gold"] if item else "",
                            "source_excerpt": source_excerpt(source, item["_gold"]) if item else "",
                        }
                        db_audit.append(audit_row)
                        if item is None:
                            continue
                        dedupe_key = (
                            hospital,
                            record["group"],
                            item["metadata"]["control_name"],
                            compact_text(item["_gold"]),
                        )
                        if dedupe_key in control_seen:
                            audit_row["status"] = "excluded"
                            audit_row["reason"] = "duplicate_control_value"
                            continue
                        control_seen.add(dedupe_key)
                        candidates.append(item)
                        included_for_record += 1
                    if included_for_record:
                        print(
                            f"  [record] group={record['group']} type={record['document_type']} controls={included_for_record}",
                            flush=True,
                        )

    auto_pool = [item for item in candidates if item["metadata"]["review_status"] == "auto_pass"]
    flagged_pool = [item for item in candidates if item["metadata"]["review_status"] == "flagged"]
    selected_auto = balanced_select(auto_pool, scaled_targets, new_target)
    selected_flagged = balanced_select(flagged_pool, {group: 20 for group in GROUP_ORDER}, min(80, len(flagged_pool)))

    selected_ids = {item["id"] for item in selected_auto} | {item["id"] for item in selected_flagged}
    for row in db_audit:
        if row.get("status") in {"auto_pass", "flagged"} and row.get("sample_id") not in selected_ids:
            row["status"] = "not_selected"
            row["reason"] = "quota_or_balance_limit"

    candidate_rows = [finalize_sample(item, system_prompt) for item in selected_auto + selected_flagged]
    auto_rows = [item for item in candidate_rows if item["metadata"]["review_status"] == "auto_pass"]
    flagged_rows = [item for item in candidate_rows if item["metadata"]["review_status"] == "flagged"]
    mixed_rows = list(auto_rows) + list(replay)
    mixed_rows.sort(key=lambda item: stable_hash("v7-mixed|" + str(item.get("id", ""))))

    write_jsonl(output_dir / "v7_single_control_candidates.jsonl", candidate_rows)
    write_jsonl(output_dir / "v7_single_control_auto_pass.jsonl", auto_rows)
    write_jsonl(output_dir / "v7_single_control_flagged.jsonl", flagged_rows)
    write_jsonl(output_dir / "v7_replay.jsonl", replay)
    write_jsonl(output_dir / "v7_train_mixed.jsonl", mixed_rows)

    audit_fields = [
        "sample_id", "hospital", "group", "setting", "document_type", "case",
        "control_name", "control_code", "status", "reason", "source_chars",
        "target_chars", "source_occurrences", "priority_score", "gold", "source_excerpt",
    ]
    with (output_dir / "v7_single_control_audit.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=audit_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(db_audit)
    with (output_dir / "v7_replay_audit.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "source_type", "status", "reason"])
        writer.writeheader()
        writer.writerows(replay_audit)

    summary = {
        "version": "v7-single-control-first-batch",
        "generated_at": "2026-11-05",
        "task": "full_record_plus_one_control_to_exact_source_value",
        "new_single_control": len(auto_rows),
        "flagged_for_review": len(flagged_rows),
        "replay": len(replay),
        "mixed_train": len(mixed_rows),
        "mix_ratio": {
            "single_control": round(len(auto_rows) / len(mixed_rows), 6) if mixed_rows else 0,
            "replay": round(len(replay) / len(mixed_rows), 6) if mixed_rows else 0,
        },
        "single_control_by_hospital": dict(Counter(item["metadata"]["hospital"] for item in auto_rows)),
        "single_control_by_group": dict(Counter(item["metadata"]["target_group"] for item in auto_rows)),
        "single_control_by_document_type": dict(Counter(item["metadata"]["document_type"] for item in auto_rows)),
        "priority_error_samples": sum(bool(item["metadata"].get("priority_from_v6_errors")) for item in auto_rows),
        "unique_single_control_cases": len({item["metadata"]["case"] for item in auto_rows}),
        "unique_controls": len({(item["metadata"]["control_name"], item["metadata"]["control_code"]) for item in auto_rows}),
        "replay_by_source": dict(Counter(item["metadata"]["source_type"] for item in replay)),
        "excluded_validation_cases": {
            "four_hospital_eval_cases": len(eval_cases),
            "four_hospital_eval_rows": sum(1 for _ in csv.DictReader(Path(args.eval_csv).open("r", encoding="utf-8-sig", newline=""))),
            "v5_validation_cases": len(load_jsonl(Path(args.v5_validation))),
            "db_records_excluded_as_validation_near_duplicate": sum(
                row.get("reason") in {"four_hospital_eval", "v5_validation"} for row in db_audit
            ),
            "replay_rows_excluded_as_validation_near_duplicate": sum(
                row.get("reason") in {"four_hospital_eval", "v5_validation"} for row in replay_audit
            ),
        },
        "quality_rules": {
            "html_removed": True,
            "deidentified": True,
            "paragraph_fields_excluded": True,
            "identity_fields_excluded": True,
            "strict_source_traceability_required": True,
            "template_placeholders_excluded": True,
            "control_code_echo_excluded": True,
            "assistant_outputs_plain_value_only": True,
            "validation_near_duplicate_filter": True,
            "max_controls_per_case": 3,
            "max_samples_per_control": 14,
        },
        "db_audit_status": dict(Counter(row.get("status", "") for row in db_audit)),
        "db_excluded_reasons": dict(Counter(row.get("reason", "") for row in db_audit if row.get("status") == "excluded")),
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
