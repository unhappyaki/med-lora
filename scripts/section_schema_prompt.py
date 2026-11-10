from __future__ import annotations


def build_section_prompt(source: str, document_type: str, schema: dict) -> str:
    document_schema = schema[document_type]
    header_fields = document_schema.get("header_fields", [])
    core_fields = document_schema.get("core_fields", [])
    optional_fields = document_schema.get("optional_fields", [])
    aliases = document_schema.get("aliases", {})
    boundary_only = document_schema.get("boundary_only_headings", [])
    alias_text = "；".join(
        f"{raw}→{canonical}"
        for raw, canonical in aliases.items()
        if raw != canonical
    )
    return (
        "你是电子病历（EMR）结构化抽取器。\n"
        f"【文书类型】{document_type}\n"
        f"【可选页眉字段】{'、'.join(header_fields) if header_fields else '无'}\n"
        f"【核心字段】{'、'.join(core_fields)}\n"
        f"【可选临床字段】{'、'.join(optional_fields) if optional_fields else '无'}\n"
        f"【同义标题归一】{alias_text if alias_text else '无'}\n\n"
        "规则：\n"
        "1. 只输出上面列出的规范字段，病历中没有明确内容的字段直接省略。\n"
        "2. slot 必须是原文中的连续文本，只取填写值，不含字段标题和冒号。\n"
        "3. 每个 section 到下一个同级标题前停止，不能吞入下一个 section。\n"
        "4. 已抽取父 section 时，不再重复拆出它内部的主诉、查体、辅助检查或人员发言。\n"
        "5. 人员姓名、日期和“某某医师：”不能被动态当成字段名。\n"
        "6. 只输出紧凑单行 JSON，格式固定为："
        '{"extractions":[{"slot":"原文值","slot_attributes":{"field_name":"规范字段名"}}]}。\n'
        f"Q: {source}\nA:"
    )


def build_section_prompt_v2(source: str, document_type: str, schema: dict) -> str:
    """Build the dedicated large-section prompt used by the v10 router.

    Version 1 was shared with mixed-granularity training.  Version 2 makes the
    task mode, canonical-name constraint, peer-heading boundary and parent/
    child exclusion explicit so the section adapter is never asked to infer
    whether it should behave like a single-control extractor.
    """
    document_schema = schema[document_type]
    header_fields = document_schema.get("header_fields", [])
    core_fields = document_schema.get("core_fields", [])
    optional_fields = document_schema.get("optional_fields", [])
    aliases = document_schema.get("aliases", {})
    boundary_only = document_schema.get("boundary_only_headings", [])
    allowed_fields = header_fields + core_fields + optional_fields
    alias_text = "；".join(
        f"{raw}→{canonical}"
        for raw, canonical in aliases.items()
        if raw != canonical
    )
    return (
        "你是电子病历（EMR）大Section边界抽取器。\n"
        "【任务模式】大Section边界抽取；不是单控件抽取。\n"
        f"【文书类型】{document_type}\n"
        f"【允许输出的规范字段名】{'、'.join(allowed_fields)}\n"
        f"【同义标题必须归一】{alias_text if alias_text else '无'}\n\n"
        f"【只作边界、禁止输出的标题】{'、'.join(boundary_only) if boundary_only else '无'}\n\n"
        "规则：\n"
        "1. field_name 只能从允许列表逐字选择；禁止照抄未归一的原始标题，禁止自造字段名。\n"
        "2. 只抽取原文中明确出现的顶层或同级Section；缺失、空白或只有标题的字段直接省略。\n"
        "3. slot 从标题后的第一个字符开始，到下一个明确的同级标题前结束；slot 必须是原文连续子串，不含标题和冒号。\n"
        "4. 父Section内部即使出现主诉、查体、辅助检查、诊断、手术指征、术式、风险、医生姓名或人员发言，也都属于父Section正文；没有独立同级标题时禁止拆成子字段。\n"
        "5. 只有原文出现独立同级标题时，才分别输出术前诊断、手术指征、拟施手术名称和方式、拟麻醉方式、术前准备、风险及注意事项等字段。\n"
        "6. 每个规范字段最多输出一次；不得重复同一值，不得把日期、人员姓名或“某某医师：”当作动态字段名。\n"
        "7. “拟诊讨论”若出现在诊断依据正文内部，仍属于诊断依据，不得另建字段。\n"
        "8. 输出应尽快闭合，只输出一行紧凑JSON，不要解释、不要Markdown。固定格式为："
        '{"extractions":[{"slot":"原文连续值","slot_attributes":{"field_name":"规范字段名"}}]}。\n'
        f"Q: {source}\nA:"
    )
