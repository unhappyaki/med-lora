from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import oracledb
from bs4 import BeautifulSoup

from medical_text_quality import (
    clean_legacy_spacing_artifacts,
    contains_spacing_artifact,
    is_symbol_only_value,
)

oracledb.init_oracle_client(
    lib_dir=r"C:\app\llx\virtual\product\12.2.0\dbhome_1\bin",
    config_dir=r"C:\app\llx\virtual\product\12.2.0\dbhome_1\network\admin",
)


SYSTEM_PROMPT = "You are a helpful assistant that responds in JSON format."
OUTPUT_REQUIREMENT = (
    "输出格式要求：只输出紧凑的单行 JSON，禁止任何缩进、换行和空格美化，"
    "禁止 ```json 代码块围栏，直接以 { 开头 } 结尾。\n"
    'JSON 根对象固定为 {"extractions": [...]}。每个元素固定使用：'
    '{"slot":"原文值","slot_attributes":{"field_name":"字段名"}}。'
)

HOSPITAL_CODES = {
    "石狮市妇幼保健院": "shishi",
    "柳城县人民医院": "liucheng",
    "邵武市中医院": "shaowu",
    "泉州市儿童医院": "quanzhou_child",
}

QUOTAS = {
    "surgery": 4,
    "outpatient": 4,
    "first_progress": 3,
    "preop": 2,
    "rescue": 2,
    "round_tcm": 1,
}

FIELD_ALIASES = {
    "手术日期": ["手术日期", "手术时间"],
    "记录时间": ["记录时间", "日期"],
    "术前诊断": ["术前诊断", "手术前诊断"],
    "术后诊断": ["术后诊断"],
    "术中诊断": ["术中诊断", "手术中诊断", "腹腔镜诊断"],
    "病理诊断": ["病理学诊断", "病理诊断"],
    "手术名称": ["手术名称", "术名称", "拟施手术名称和方式", "手术名称和方式"],
    "手术者": ["手术者", "手术人员", "术者"],
    "助手": ["助手"],
    "麻醉方式": ["麻醉方式", "麻醉方法", "拟施麻醉方式"],
    "麻醉医师": ["麻醉师", "麻醉者", "麻醉人员"],
    "手术经过": ["手术经过", "手术步骤", "手术操作记录", "手术记录"],
    "出血量": ["出血量", "术中出血量"],
    "标本": ["标本", "送检标本"],
    "记录者": ["记录者", "签名"],
    "就诊日期": ["就诊日期"],
    "门诊号": ["门诊号", "病历号"],
    "住院号": ["住院号"],
    "病案号": ["病案号"],
    "床号": ["床号", "病床号"],
    "患者姓名": ["患者姓名", "姓名"],
    "性别": ["性别"],
    "年龄": ["年龄"],
    "民族": ["民族"],
    "家庭住址": ["家庭住址", "住址"],
    "联系电话": ["联系电话", "手机号码", "联系电话号码"],
    "身份证号码": ["身份证号码", "身份证号"],
    "药物过敏史": ["药物过敏史", "过敏史"],
    "病史陈述者": ["病史陈述者"],
    "主诉": ["主诉"],
    "现病史": ["现病史"],
    "既往史": ["既往史"],
    "个人史": ["个人史"],
    "查体": ["入院查体", "查体", "体格检查"],
    "专科情况": ["专科情况", "专科检查"],
    "辅助检查": ["辅助检查", "入院检查"],
    "诊断": [
        "诊断",
        "中医诊断",
        "门诊诊断",
        "初步诊断",
        "目前诊断考虑",
        "目前诊断",
    ],
    "处理意见": ["处理意见", "处理", "治疗意见", "诊疗意见"],
    "病历特点": ["病历特点", "病例特点"],
    "诊断依据": ["诊断依据"],
    "鉴别诊断": ["鉴别诊断"],
    "诊疗计划": ["诊疗计划", "治疗计划"],
    "病情摘要": ["病情摘要", "病历摘要", "简要病情"],
    "手术指征": ["手术指征", "手术适应证", "手术适应症"],
    "术前准备": ["术前准备"],
    "注意事项": ["注意事项"],
    "医师签名": ["医师签名", "经治医师", "住院医师"],
    "术者签名": ["术者签名", "术者签名/经治医师签名"],
    "讨论日期": ["讨论日期", "讨论时间"],
    "讨论地点": ["讨论地点"],
    "主持人": ["主持人", "抢救主持人", "主持人签名"],
    "参加讨论人员": ["参加讨论者姓名及职称", "参加讨论人员", "参加人员"],
    "讨论内容": ["讨论内容", "讨论记录"],
    "讨论结论": ["讨论结论", "主持人小结意见"],
    "抢救开始时间": ["抢救开始时间", "抢救开始日期时间"],
    "抢救结束时间": ["抢救结束时间", "抢救结束日期时间"],
    "参加抢救人员": ["参加抢救人员", "抢救人员及职称", "参加抢救人员名单"],
    "参加抢救最高职称医师": ["参加抢救最高职称医师"],
    "病情变化情况": ["病情变化情况"],
    "抢救措施": ["抢救措施", "抢救经过描述"],
    "抢救结果": ["抢救结果", "抢救是否成功"],
    "中医辨病辨证依据": [
        "中医辨病辨证依据",
        "中医辨病辨证依据与鉴别诊断",
        "中医辨病辨证依据及鉴别诊断",
        "中医辨病辨证依据与诊断诊断",
        "中医辨病辨证依据与诊断",
    ],
    "类证鉴别": ["类证鉴别"],
    "证候": ["证候", "症候", "中医证候"],
    "中医治则": ["中医治则", "治则治法"],
    "西医诊断": ["西医诊断"],
    "主治医师查房意见": [
        "主治医师查房意见",
        "主治医师查房指出",
        "主治医师查房",
        "某某主治医师查房指出",
        "某某主治医师查房",
        "查房意见",
    ],
}

# These labels are not training targets, but they still delimit the preceding
# field.  Without them, a useful field such as 麻醉人员 or 标本 can absorb an
# unrelated block from a structured operation-note template.
BOUNDARY_ONLY_ALIASES = {
    "就诊科别",
    "婚姻",
    "职业",
    "医嘱",
    "备注",
    "医生",
    "拟诊讨论",
    "抢救类型",
    "举宫器",
    "气针情况",
    "压力",
    "进气量",
    "套针情况",
    "加入评分9宫格",
    "皮下气肿",
    "其它",
    "其他",
    "西医辨病辨证依据与鉴别诊断",
    "西医辨病辨证依据及鉴别诊断",
    "诊断及鉴别诊断",
    "术者查看患者",
    "术者查看病人",
    "术者查看病人后指示",
}

DATE_FIELDS = {
    "手术日期",
    "记录时间",
    "就诊日期",
    "讨论日期",
    "抢救开始时间",
    "抢救结束时间",
}

SURGERY_BOILERPLATE_PATTERNS = (
    r"^[（(]?记录不够时以病历记录单继续记录附于后面[）)]?[：:]?\s*",
    r"^[（(]?包括病人体位、切口处理经过及病理标本肉眼所见等[）)]?[：:]?\s*",
)

IDENTITY_FIELDS = {
    "患者姓名": "患者甲",
    "家庭住址": "某省某市某区",
    "联系电话": "13800000000",
    "身份证号码": "350000000000000000",
    "门诊号": "ID000001",
    "住院号": "ID000001",
    "病案号": "ID000001",
    "床号": "00床",
    "手术者": "医师甲",
    "助手": "医师乙",
    "麻醉医师": "医师丙",
    "记录者": "医师甲",
    "医师签名": "医师甲",
    "术者签名": "医师甲",
    "主持人": "医师甲",
    "参加讨论人员": "医师甲、医师乙",
    "参加抢救人员": "医师甲、医师乙",
    "参加抢救最高职称医师": "医师甲",
}

# These labels commonly appear as subheadings inside a larger summary field.
# They must remain part of the parent value instead of prematurely ending it.
NESTED_FIELD_BOUNDARIES = {
    "病历特点": {
        "主诉",
        "现病史",
        "既往史",
        "个人史",
        "查体",
        "专科情况",
    },
    "诊断依据": {
        "主诉",
        "现病史",
        "既往史",
        "个人史",
        "查体",
        "专科情况",
        "辅助检查",
    },
}

NARRATIVE_FIELDS = {
    "主诉",
    "现病史",
    "既往史",
    "个人史",
    "查体",
    "专科情况",
    "辅助检查",
    "处理意见",
    "病历特点",
    "诊断依据",
    "鉴别诊断",
    "诊疗计划",
    "病情摘要",
    "手术指征",
    "术前准备",
    "注意事项",
    "讨论内容",
    "讨论结论",
    "病情变化情况",
    "抢救措施",
    "抢救结果",
    "中医辨病辨证依据",
    "类证鉴别",
    "中医治则",
    "主治医师查房意见",
    "手术经过",
}

TEMPLATE_PLACEHOLDER_RE = re.compile(
    r"\{(?:体温|脉搏|脉博|呼吸|呼吸频率|收缩压|舒张压|身高|体重|年龄阶段)[^}]*\}"
)

REAL_SIGNER_RE = re.compile(
    r"(?:医师签名|主持人签名|记录者|某某医师)[ \t]*[：:][ \t]*(?:\n[ \t]*)?"
    r"(?!医师[甲乙丙丁戊己](?:[ \t]*(?:\n|$))|某某|术者签名|医师签名|记录时间|记录者|主持人签名)"
    r"[\u4e00-\u9fff]{2,4}(?=[ \t]*(?:\n|$))"
)

SYSTEM_FIELD_NAMES = {
    "序列号",
    "最后修改时间",
    "病人id",
    "病人ID",
    "病案号",
    "住院号",
    "床号",
    "住院次数",
    "住院记录表编号",
    "科室代码",
    "科室编码",
    "科室名称",
    "病区名称",
    "病房号",
    "病床号",
}

GROUP_DOCUMENT_TYPES = {
    "surgery": ("手术记录",),
    "first_progress": ("首次病程记录",),
    "preop": ("术前小结", "术前讨论", "术前讨论记录"),
    "rescue": ("抢救记录", "死亡抢救记录"),
}

GROUP_REQUIRED = {
    "surgery": ({"手术名称", "术前诊断", "术后诊断", "术中诊断"}, {"手术经过"}),
    "outpatient": ({"主诉"}, {"现病史"}),
    "first_progress": ({"病历特点"}, {"诊疗计划", "诊断", "诊断依据"}),
    "preop": ({"术前诊断", "病情摘要"}, {"手术名称", "讨论内容"}),
    "rescue": ({"抢救措施", "病情变化情况"}, {"抢救结果", "抢救开始时间", "参加抢救人员"}),
    "round_tcm": ({"主治医师查房意见", "中医辨病辨证依据", "诊断依据"}, {"诊断", "西医诊断", "中医治则"}),
}

MIN_EXTRACTIONS = {
    "surgery": 5,
    "outpatient": 6,
    "first_progress": 3,
    "preop": 5,
    "rescue": 3,
    "round_tcm": 3,
}

FIELD_MAX_CHARS = {
    "患者姓名": 20,
    "性别": 10,
    "年龄": 20,
    "民族": 20,
    "家庭住址": 100,
    "联系电话": 30,
    "身份证号码": 30,
    "门诊号": 50,
    "住院号": 50,
    "病案号": 50,
    "床号": 20,
    "就诊日期": 60,
    "手术日期": 60,
    "记录时间": 60,
    "讨论日期": 60,
    "抢救开始时间": 60,
    "抢救结束时间": 60,
    "手术者": 120,
    "助手": 120,
    "麻醉医师": 120,
    "记录者": 120,
    "医师签名": 120,
    "术者签名": 120,
    "主持人": 120,
    "参加讨论人员": 250,
    "参加抢救人员": 250,
    "参加抢救最高职称医师": 120,
    "麻醉方式": 80,
    "出血量": 100,
    "标本": 250,
    "手术名称": 300,
    "诊断": 500,
    "术前诊断": 500,
    "术后诊断": 500,
    "术中诊断": 500,
    "病理诊断": 500,
    "辅助检查": 1000,
    "注意事项": 800,
    "术前准备": 1000,
}


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def valid_field_length(field_name: str, value: str) -> bool:
    return 0 < len(value) <= FIELD_MAX_CHARS.get(field_name, 3500)


def read_lob(value) -> str:
    if value is None:
        return ""
    if hasattr(value, "read"):
        value = value.read()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def clean_rich_text(value: str) -> str:
    if not value:
        return ""
    soup = BeautifulSoup(value, "lxml")
    for element in soup(["script", "style", "svg"]):
        element.decompose()
    text = soup.get_text(separator="\n")
    text = clean_legacy_spacing_artifacts(html.unescape(text))
    lines = []
    for line in text.splitlines():
        line = re.sub(r"[\t ]+", " ", line).strip()
        if not line:
            continue
        if lines and line == lines[-1]:
            continue
        lines.append(line)
    return "\n".join(lines)


def choose_content(values: list) -> str:
    candidates = [clean_rich_text(read_lob(value)) for value in values]
    return max(candidates, key=len, default="")


def general_deidentify(text: str, patient_name: str, doctor_names: list[str]) -> str:
    replacements = []
    if patient_name and len(patient_name.strip()) >= 2:
        replacements.append((patient_name.strip(), "患者甲"))
    for index, name in enumerate(dict.fromkeys(doctor_names), start=1):
        name = name.strip()
        if len(name) >= 2:
            replacements.append((name, f"医师{'甲乙丙丁戊己'[min(index - 1, 5)]}"))
    for source, target in sorted(replacements, key=lambda item: -len(item[0])):
        text = text.replace(source, target)
    text = re.sub(r"(?<!\d)1[3-9]\d{9}(?!\d)", "13800000000", text)
    text = re.sub(
        r"(?<![0-9A-Za-z])\d{17}[0-9Xx](?![0-9A-Za-z])",
        "350000000000000000",
        text,
    )
    protected_title_prefixes = {
        "主持人",
        "记录者",
        "手术者",
        "麻醉者",
        "麻醉人员",
        "抢救主持人",
    }
    text = re.sub(
        r"([\u4e00-\u9fff]{2,4})"
        r"(麻醉科主任|科主任|主任医师|副主任医师|主治\s*医师|住院\s*医师|医师|医生|麻醉师|护士)",
        lambda match: (
            match.group(0)
            if match.group(1) in protected_title_prefixes
            else "某某" + re.sub(r"\s+", "", match.group(2))
        ),
        text,
    )
    # Patient names occasionally differ between the EMR index and the rich
    # text (newborn “某某之子/女” records are common).  Remove names that are
    # explicitly followed by a sex marker without touching ordinary prose.
    text = re.sub(
        r"(?:患者|患儿)\s*[\u4e00-\u9fff]{2,4}(?:之子|之女)?"
        r"(?=\s*[,，、]\s*(?:男|女))",
        "患者甲",
        text,
    )
    text = re.sub(
        r"([：:])\s*[\u4e00-\u9fff]{2,4}(?=\s+(?:第一助手|第二助手))",
        r"\1医师甲",
        text,
    )
    return text


def spaced_label_pattern(alias: str) -> str:
    """Match labels whose characters may be separated by horizontal spaces."""
    return r"[ \t]*".join(re.escape(character) for character in alias)


def alias_matches(source: str) -> list[dict]:
    matches = []
    aliases_by_field = list(FIELD_ALIASES.items()) + [
        (None, sorted(BOUNDARY_ONLY_ALIASES, key=len, reverse=True))
    ]
    for canonical, aliases in aliases_by_field:
        for alias in sorted(aliases, key=len, reverse=True):
            label_pattern = spaced_label_pattern(alias)
            pattern = re.compile(
                rf"(?m)(?:(?<!\S)|(?<=[：:；;。.!?、．])){label_pattern}"
                rf"(?:[ \t]*[：:][ \t]*|[ \t]*\r?\n[ \t]*[：:][ \t]*|[ \t]*(?=\r?$))"
            )
            for match in pattern.finditer(source):
                if canonical == "药物过敏史":
                    prefix = source[max(0, match.start() - 12) : match.start()]
                    if re.search(r"(?:否认|无)(?:食物[、,，])?$", prefix):
                        continue
                matches.append(
                    {
                        "canonical": canonical,
                        "alias": alias,
                        "start": match.start(),
                        "value_start": match.end(),
                    }
                )
    deduplicated = {}
    for match in matches:
        key = (match["start"], match["value_start"])
        previous = deduplicated.get(key)
        if previous is None or len(match["alias"]) > len(previous["alias"]):
            deduplicated[key] = match
    return sorted(deduplicated.values(), key=lambda item: (item["start"], item["value_start"]))


def is_template_date(field_name: str, value: str) -> bool:
    if field_name not in DATE_FIELDS:
        return False
    compact = re.sub(r"\s+", "", value)
    if re.search(r"时间日期|YYYY|MM|DD|年\s*月\s*日", compact, re.I):
        return True
    if field_name == "手术日期" and re.search(r"时分|时至|分至", compact):
        return True
    if field_name == "手术日期":
        # A duration such as “5小时” is not an operation date.  Broken-line
        # dates remain valid after whitespace removal (for example 2021-04-2\n2).
        return not bool(
            re.search(
                r"(?:19|20)\d{2}(?:年|[-./])\d{1,2}(?:月|[-./])\d{1,2}",
                compact,
            )
        )
    return not bool(
        re.search(r"(?:19|20)\d{2}", compact)
        or re.search(r"(?<!\d)\d{1,2}:\d{1,2}(?!\d)", compact)
        or re.search(r"\d{1,2}时\d{1,2}分", compact)
    )


def clean_extracted_value(field_name: str, value: str) -> str:
    value = value.strip(" \n\r\t：:，,；;")
    replacement = IDENTITY_FIELDS.get(field_name)
    if replacement and replacement in value:
        # A malformed document can place the next field immediately after a
        # signer with no whitespace (for example “医师甲查体：…”).  The
        # deidentified placeholder is still an exact source substring and is
        # the only safe supervision target for an identity field.
        return replacement
    if field_name == "手术经过":
        for pattern in SURGERY_BOILERPLATE_PATTERNS:
            value = re.sub(pattern, "", value, count=1)
        # Some source templates append an unlabeled signer/date to the final
        # narrative.  Keep the operation narrative but do not supervise the
        # model to include the signature in 手术经过.
        value = re.sub(
            r"\n(?:医师[甲乙丙丁戊己]|某某医师)[：:]?"
            r"(?:\n\{?(?:19|20)\d{2}[^\n]{0,30}\}?)?\s*$",
            "",
            value,
        )
    if field_name == "抢救结果":
        value = re.sub(
            r"\n主持人(?:医师[甲乙丙丁戊己]|某某医师)"
            r"(?:\n医师签名[：:]?(?:\n医师[甲乙丙丁戊己])?)?\s*$",
            "",
            value,
        )
    if field_name in NARRATIVE_FIELDS:
        value = re.sub(
            r"\n(?:某某医师|医师[甲乙丙丁戊己])\s*[：:]?\s*$",
            "",
            value,
        )
        value = re.sub(
            r"(?:\n|[。；;])\s*(?:\d{1,2}|[一二三四五六七八九十]{1,3})"
            r"\s*[、.．]\s*$",
            "",
            value,
        )
    return value.strip(" \n\r\t：:，,；;")


def is_unusable_template_value(field_name: str, value: str) -> bool:
    compact = re.sub(r"\s+", "", value).strip("，,。；;：:")
    if not compact:
        return True
    if compact in {
        "{有无}",
        "{病历陈述者}",
        "{麻醉方式}",
        "{年龄阶段}",
        "。",
    }:
        return True
    if re.fullmatch(r"[{}（）()\[\]<>《》/._-]+", compact):
        return True
    if field_name == "病情摘要" and re.fullmatch(
        r"患者因[“\"']?[”\"']?入院", compact
    ):
        return True
    if field_name in {"查体", "辅助检查"}:
        without_placeholders = re.sub(r"\{[^{}]{1,30}\}", "", compact)
        if compact.count("{") >= 3 and len(without_placeholders) < 35:
            return True
    return False


def sanitize_labeled_identities(source: str) -> str:
    matches = alias_matches(source)
    replacements = []
    for index, match in enumerate(matches):
        canonical = match["canonical"]
        if canonical is None or canonical not in IDENTITY_FIELDS:
            continue
        end = matches[index + 1]["start"] if index + 1 < len(matches) else len(source)
        raw = source[match["value_start"] : end]
        left = len(raw) - len(raw.lstrip(" \n：:"))
        right = len(raw.rstrip())
        value_start = match["value_start"] + left
        value_end = match["value_start"] + right
        if value_start >= value_end:
            continue
        if value_end - value_start > 200:
            continue
        replacements.append(
            (value_start, value_end, IDENTITY_FIELDS[canonical])
        )
    for start, end, replacement in sorted(replacements, reverse=True):
        source = source[:start] + replacement + source[end:]
    return source


def extract_labeled_fields(source: str) -> list[dict]:
    matches = alias_matches(source)
    extractions = []
    nested_match_indexes = set()
    for index, match in enumerate(matches):
        if match["canonical"] is None:
            continue
        if index in nested_match_indexes:
            continue
        ignored_boundaries = NESTED_FIELD_BOUNDARIES.get(match["canonical"], set())
        end = len(source)
        for following_index, following in enumerate(
            matches[index + 1 :], start=index + 1
        ):
            if following["canonical"] in ignored_boundaries:
                nested_match_indexes.add(following_index)
                continue
            end = following["start"]
            break
        raw = source[match["value_start"] : end]
        value = clean_extracted_value(match["canonical"], raw)
        if not value:
            continue
        if is_unusable_template_value(match["canonical"], value):
            continue
        if is_template_date(match["canonical"], value):
            continue
        if not valid_field_length(match["canonical"], value):
            continue
        offset = source.find(value, match["value_start"], end)
        if offset < 0:
            continue
        extractions.append(
            {
                "field_name": match["canonical"],
                "text": value,
                "start": offset,
                "method": "label_boundary",
            }
        )
    return extractions


def normalize_field_name(value: str) -> str:
    value = re.sub(r"\s+", "", str(value or "")).strip("：:")
    reverse = {
        alias: canonical
        for canonical, aliases in FIELD_ALIASES.items()
        for alias in aliases
    }
    value = reverse.get(value, value)
    if value == "病例特点":
        value = "病历特点"
    if value in {"治则治法", "中医治则治法"}:
        value = "中医治则"
    if value in {"参加抢救人员名单", "抢救人员及职称"}:
        value = "参加抢救人员"
    return value


def sharing_columns(cursor) -> set[str]:
    cursor.execute(
        "select column_name from user_tab_columns where table_name='EMR_Z_RECORDFORSHARING'"
    )
    return {row[0] for row in cursor}


def load_sharing_fields(cursor, record_id: str, category_id: str, has_name: bool) -> list[tuple[str, str]]:
    sharing_name = "r.mingcheng" if has_name else "null"
    cursor.execute(
        f"""
        select nvl({sharing_name}, nvl(x.mingcheng, nvl(r.field, nvl(x.field, r.de_code)))) field_name,
               r.value
        from emr_z_recordforsharing r
        left join emr_x_sharingcode x
          on x.fenleiid=:category_id
         and (x.de_code=r.de_code or (x.field=r.field and x.field is not null))
        where r.bingliid=:record_id
          and r.value is not null
          and length(trim(r.value)) > 0
        """,
        category_id=category_id,
        record_id=record_id,
    )
    output = []
    seen = set()
    for field_name, value in cursor:
        name = normalize_field_name(field_name)
        text = str(value or "").strip()
        key = (name, text)
        if (
            not name
            or name.startswith("段落")
            or name in SYSTEM_FIELD_NAMES
            or not text
            or key in seen
        ):
            continue
        seen.add(key)
        output.append(key)
    return output


def merge_extractions(source: str, labeled: list[dict], sharing: list[tuple[str, str]], patient_name: str, doctor_names: list[str]) -> list[dict]:
    merged = list(labeled)
    for field_name, raw_value in sharing:
        value = general_deidentify(raw_value, patient_name, doctor_names).strip()
        if field_name in IDENTITY_FIELDS:
            value = IDENTITY_FIELDS[field_name]
        if not valid_field_length(field_name, value):
            continue
        start = source.find(value)
        if start < 0:
            continue
        merged.append(
            {
                "field_name": field_name,
                "text": value,
                "start": start,
                "method": "recordforsharing",
            }
        )
    seen = set()
    output = []
    for item in sorted(merged, key=lambda row: (row["start"], row["field_name"], -len(row["text"]))):
        key = (item["field_name"], item["text"])
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def has_excessive_repetition(value: str) -> bool:
    compact = re.sub(r"\s+", "", value)
    for size in range(3, 9):
        for start in range(max(0, len(compact) - 120)):
            token = compact[start : start + size]
            if (
                len(token) == size
                and len(set(token)) > 1
                and compact.count(token) >= 5
            ):
                return True
    return False


def normalized_core(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]", "", value)


def has_duplicated_surgery_name(value: str) -> bool:
    compact = re.sub(r"\s+", "", value).strip("，,。；;")
    if re.search(r"(.{4,20})\1$", compact):
        return True
    return bool(re.search(r"术[、,，][^。；;]{2,20}术$", compact))


def group_quality_valid(group: str, source: str, extractions: list[dict]) -> bool:
    values = defaultdict(list)
    for item in extractions:
        values[item["field_name"]].append(item["text"])
    all_values = [item["text"] for item in extractions]
    if any(contains_spacing_artifact(value) for value in all_values):
        return False
    if any(is_symbol_only_value(value) for value in all_values):
        return False
    if any(TEMPLATE_PLACEHOLDER_RE.search(value) for value in all_values):
        return False
    if REAL_SIGNER_RE.search(source):
        return False
    if group == "outpatient":
        chief = "".join(values["主诉"])
        present = "".join(values["现病史"])
        chief_core = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]", "", chief)
        present_core = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]", "", present)
        if len(chief_core) < 4 or len(chief_core) > 60 or len(present_core) < 6:
            return False
        if re.fullmatch(r"\d+", present_core):
            return False
        if has_excessive_repetition(chief) or has_excessive_repetition(present):
            return False
        if any(
            "八闽健康码" in value
            or (
                value.count("具体情况描述") >= 2
                and "无" in value
                and "有" in value
            )
            for value in values["辅助检查"]
        ):
            return False
        age_match = re.search(r"(\d{1,3})岁", "".join(values["年龄"]))
        if age_match and int(age_match.group(1)) >= 18 and "患儿" in present:
            return False
    if group == "first_progress":
        if len(values["鉴别诊断"]) > 1 or len(values["诊疗计划"]) > 1:
            return False
        if not values["诊疗计划"] or not (values["诊断"] or values["西医诊断"]):
            return False
        combined = "\n".join(all_values)
        if "{年龄阶段}" in combined or "岁岁" in combined:
            return False
        diagnoses = "".join(values["诊断"] + values["西医诊断"])
        if "阴茎" in diagnoses and (
            any(value.strip() == "女" for value in values["性别"])
            or re.search(r"(?:患儿|患者)[^。；;]{0,20}女", source)
        ):
            return False
    if group == "preop":
        diagnoses = {
            re.sub(r"\s+", "", value).strip("，,。；;")
            for value in values["术前诊断"]
            if value.strip()
        }
        if len(diagnoses) > 1:
            return False
        if any(re.sub(r"\s+", "", value).rstrip("。；;").endswith("麻") for value in values["麻醉方式"]):
            return False
    if group == "round_tcm":
        tcm_basis = "".join(values["中医辨病辨证依据"])
        has_tcm = (
            len(normalized_core(tcm_basis)) >= 40
            and bool(values["中医治则"] or values["证候"])
        )
        round_text = "".join(values["主治医师查房意见"])
        has_round = len(normalized_core(round_text)) >= 50 and len(extractions) >= 4
        if not (has_tcm or has_round):
            return False
    if group == "surgery":
        if any(has_duplicated_surgery_name(value) for value in values["手术名称"]):
            return False
        if any(re.match(r"手\s*\n", value) for value in values["术中诊断"]):
            return False
    return True


def group_valid(group: str, source: str, extractions: list[dict]) -> bool:
    names = {item["field_name"] for item in extractions}
    if len(extractions) < MIN_EXTRACTIONS[group]:
        return False
    first, second = GROUP_REQUIRED[group]
    return (
        bool(names & first)
        and bool(names & second)
        and group_quality_valid(group, source, extractions)
    )


def old_source_hashes(path: Path) -> set[str]:
    if not path.exists():
        return set()
    hashes = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        user = row["messages"][1]["content"]
        source = user.split("Q: ", 1)[-1].rsplit("\nA:", 1)[0]
        hashes.add(stable_hash(re.sub(r"\s+", "", source)))
    return hashes


def query_inpatient(cursor, document_types: tuple[str, ...], limit: int = 250) -> list[dict]:
    binds = {f"doc_{index}": value for index, value in enumerate(document_types)}
    placeholders = ",".join(f":doc_{index}" for index in range(len(document_types)))
    cursor.execute(
        f"""
        select * from (
          select b.bingliid, b.xingming, b.jzys, b.zzys, b.zrys, b.sxys,
                 b.fenleiid, f.mingcheng,
                 mx.jzneirong, mx.zzneirong, mx.zrneirong, mx.gdneirong, mx.sxneirong
          from emr_z_bingli b
          join emr_z_binglimx mx on mx.bingliid=b.bingliid
          join emr_z_blfenlei f on f.fenleiid=b.fenleiid
          where f.mingcheng in ({placeholders})
          order by b.blshijian desc nulls last
        ) where rownum <= {int(limit)}
        """,
        binds,
    )
    rows = []
    for row in cursor:
        rows.append(
            {
                "record_id": str(row[0]),
                "patient_name": str(row[1] or ""),
                "doctor_names": [str(value or "") for value in row[2:6]],
                "category_id": str(row[6]),
                "document_type": str(row[7]),
                "content": choose_content(list(row[8:13])),
                "setting": "inpatient",
            }
        )
    return rows


def query_rounds(cursor, limit: int = 250) -> list[dict]:
    cursor.execute(
        f"""
        select * from (
          select b.bingliid, b.xingming, b.jzys, b.zzys, b.zrys, b.sxys,
                 b.fenleiid, f.mingcheng,
                 mx.jzneirong, mx.zzneirong, mx.zrneirong, mx.gdneirong, mx.sxneirong
          from emr_z_bingli b
          join emr_z_binglimx mx on mx.bingliid=b.bingliid
          join emr_z_blfenlei f on f.fenleiid=b.fenleiid
          where f.mingcheng like '%主治医师查房%'
             or f.mingcheng like '%中医%'
          order by b.blshijian desc nulls last
        ) where rownum <= {int(limit)}
        """
    )
    rows = []
    for row in cursor:
        rows.append(
            {
                "record_id": str(row[0]),
                "patient_name": str(row[1] or ""),
                "doctor_names": [str(value or "") for value in row[2:6]],
                "category_id": str(row[6]),
                "document_type": str(row[7]),
                "content": choose_content(list(row[8:13])),
                "setting": "inpatient",
            }
        )
    return rows


def query_outpatient(cursor, limit: int = 400) -> list[dict]:
    cursor.execute(
        "select column_name from user_tab_columns where table_name='EMR_M_LCBINGLIMX'"
    )
    columns = {row[0] for row in cursor}
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
    rows = []
    for row in cursor:
        rows.append(
            {
                "record_id": str(row[0]),
                "patient_name": str(row[1] or ""),
                "doctor_names": [str(row[2] or "")],
                "category_id": str(row[3]),
                "document_type": str(row[4]),
                "content": choose_content(list(row[5:7])),
                "setting": "outpatient",
            }
        )
    return rows


def build_prompt(prompt_v5: str, source: str) -> str:
    return f"{prompt_v5.rstrip()}\n\n{OUTPUT_REQUIREMENT}\n\nQ: {source}\nA:"


def remove_source_boilerplate(source: str) -> str:
    source = re.sub(
        r"(?m)^[（(]?记录不够时以病历记录单继续记录附于后面[）)]?[：:]?\s*$",
        "",
        source,
    )
    source = re.sub(
        r"(?m)^[（(]?包括病人体位、切口处理经过及病理标本肉眼所见等[）)]?[：:]?\s*$",
        "",
        source,
    )
    return re.sub(r"\n{3,}", "\n\n", source).strip()


def process_record(cursor, hospital: str, group: str, record: dict, has_sharing_name: bool, prompt_v5: str, old_hashes: set[str], batch_hashes: set[str]) -> tuple[dict | None, dict]:
    audit = {
        "hospital": hospital,
        "group": group,
        "setting": record["setting"],
        "document_type": record["document_type"],
        "status": "excluded",
        "reason": "",
        "source_chars": 0,
        "extraction_count": 0,
        "fields": "",
    }
    source = record["content"]
    if len(source) < 80:
        audit["reason"] = "source_too_short"
        return None, audit
    source = general_deidentify(
        source, record["patient_name"], record["doctor_names"]
    )
    source = sanitize_labeled_identities(source)
    source = remove_source_boilerplate(source)
    audit["source_chars"] = len(source)
    if re.search(r"<\s*/?\s*[A-Za-z_][^>]*>", source):
        audit["reason"] = "html_tag_residual"
        return None, audit
    source_hash = stable_hash(re.sub(r"\s+", "", source))
    if source_hash in old_hashes:
        audit["reason"] = "duplicate_v5"
        return None, audit
    if source_hash in batch_hashes:
        audit["reason"] = "duplicate_batch"
        return None, audit
    labeled = extract_labeled_fields(source)
    sharing = load_sharing_fields(
        cursor,
        record["record_id"],
        record["category_id"],
        has_sharing_name,
    )
    extractions = merge_extractions(
        source,
        labeled,
        sharing,
        record["patient_name"],
        record["doctor_names"],
    )
    extractions = [
        item
        for item in extractions
        if item["text"] in source
        and valid_field_length(item["field_name"], item["text"])
        and item["field_name"] not in SYSTEM_FIELD_NAMES
        and not item["field_name"].startswith("段落")
    ]
    audit["extraction_count"] = len(extractions)
    audit["fields"] = "|".join(item["field_name"] for item in extractions)
    if not group_valid(group, source, extractions):
        audit["reason"] = "insufficient_required_fields"
        return None, audit
    extraction_signature = stable_hash(
        group
        + "|"
        + json.dumps(
            [
                (item["field_name"], re.sub(r"\s+", "", item["text"]))
                for item in extractions
                if item["field_name"] not in IDENTITY_FIELDS
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    extraction_key = "extraction:" + extraction_signature
    if extraction_key in batch_hashes:
        audit["reason"] = "duplicate_extractions"
        return None, audit
    if record["patient_name"] and record["patient_name"] in source:
        audit["reason"] = "patient_name_not_deidentified"
        return None, audit
    extraction_payload = [
        {
            "slot": item["text"],
            "slot_attributes": {"field_name": item["field_name"]},
        }
        for item in extractions
    ]
    public_id = (
        f"v6db-{HOSPITAL_CODES[hospital]}-{group}-"
        f"{stable_hash(hospital + '|' + record['record_id'])[:16]}"
    )
    assistant = json.dumps(
        {"extractions": extraction_payload},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    sample = {
        "id": public_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(prompt_v5, source)},
            {"role": "assistant", "content": assistant},
        ],
        "metadata": {
            "case_id": public_id,
            "source_hospital": hospital,
            "source_setting": record["setting"],
            "document_type": record["document_type"],
            "target_group": group,
            "split": "candidate",
            "review_status": "pending",
            "hard_case": True,
            "source_chars": len(source),
            "target_chars": len(assistant),
            "extraction_count": len(extraction_payload),
            "label_methods": dict(Counter(item["method"] for item in extractions)),
            "deidentified": True,
            "html_removed": True,
        },
    }
    batch_hashes.add(source_hash)
    batch_hashes.add(extraction_key)
    audit["status"] = "included"
    audit["reason"] = ""
    audit["sample_id"] = public_id
    return sample, audit


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--old-v5", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    prompt_v5 = Path(args.prompt).read_text(encoding="utf-8").strip()
    old_hashes = old_source_hashes(Path(args.old_v5))
    connections = json.loads(os.environ["V6_CONNECTIONS_JSON"])
    samples = []
    audits = []
    batch_hashes: set[str] = set()
    coverage = defaultdict(Counter)

    for hospital in sorted(connections):
        print(f"Preparing {hospital}...", flush=True)
        with oracledb.connect(**connections[hospital]) as database:
            with database.cursor() as cursor:
                has_sharing_name = "MINGCHENG" in sharing_columns(cursor)
                grouped_candidates = {
                    group: query_inpatient(cursor, document_types)
                    for group, document_types in GROUP_DOCUMENT_TYPES.items()
                }
                grouped_candidates["outpatient"] = query_outpatient(cursor)
                grouped_candidates["round_tcm"] = query_rounds(cursor)
                for group, records in grouped_candidates.items():
                    records = sorted(
                        records,
                        key=lambda item: stable_hash(
                            hospital + "|" + group + "|" + item["record_id"]
                        ),
                    )
                    selected = 0
                    for record in records:
                        sample, audit = process_record(
                            cursor,
                            hospital,
                            group,
                            record,
                            has_sharing_name,
                            prompt_v5,
                            old_hashes,
                            batch_hashes,
                        )
                        audits.append(audit)
                        if sample is None:
                            continue
                        samples.append(sample)
                        selected += 1
                        coverage[hospital][group] += 1
                        if selected >= QUOTAS[group]:
                            break

    samples.sort(key=lambda item: item["id"])
    write_jsonl(output_dir / "v6_supplement_candidates.jsonl", samples)
    audit_fields = [
        "sample_id",
        "hospital",
        "group",
        "setting",
        "document_type",
        "status",
        "reason",
        "source_chars",
        "extraction_count",
        "fields",
    ]
    with (output_dir / "v6_supplement_audit.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=audit_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(audits)
    summary = {
        "version": "v6-first-batch",
        "review_status": "pending_human_review",
        "candidate_cases": len(samples),
        "by_hospital": {
            hospital: dict(counts) for hospital, counts in sorted(coverage.items())
        },
        "by_group": dict(
            Counter(sample["metadata"]["target_group"] for sample in samples)
        ),
        "by_document_type": dict(
            Counter(sample["metadata"]["document_type"] for sample in samples)
        ),
        "total_extractions": sum(
            sample["metadata"]["extraction_count"] for sample in samples
        ),
        "deidentified": True,
        "html_removed": True,
        "v5_duplicate_sources_excluded": sum(
            row["reason"] == "duplicate_v5" for row in audits
        ),
        "excluded_reasons": dict(
            Counter(row["reason"] for row in audits if row["status"] != "included")
        ),
        "quotas_per_hospital": QUOTAS,
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        "# V6 第一批数据库补训候选数据\n\n"
        "本批数据来自四家医院开发库，已做去标识化、原文精确对齐和 V5 重复检查。\n"
        "当前状态为 pending_human_review；人工抽查通过后再合并到正式 V6 训练集。\n\n"
        "- v6_supplement_candidates.jsonl：候选 SFT 数据\n"
        "- v6_supplement_audit.csv：纳入/排除审计\n"
        "- dataset_summary.json：数量与覆盖统计\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
