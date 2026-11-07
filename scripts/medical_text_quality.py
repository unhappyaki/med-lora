from __future__ import annotations

import re


HTML_SPACING_ARTIFACT_RE = re.compile(
    r"&(?:nbsp|#160|#x0*a0);|&[\u00a0\u2007\u202f]",
    re.IGNORECASE,
)
MEANINGFUL_CONTENT_RE = re.compile(r"[0-9A-Za-z\u4e00-\u9fff]")


def clean_legacy_spacing_artifacts(value: str) -> str:
    """Remove HTML spacing residue without rewriting clinical text."""
    text = str(value or "")
    if not re.search(
        r"&(?:nbsp|#160|#x0*a0);|[\u00a0\u2007\u202f\u3000]",
        text,
        re.IGNORECASE,
    ):
        return text
    text = re.sub(
        r"&(?:nbsp|#160|#x0*a0);",
        "\u00a0",
        text,
        flags=re.IGNORECASE,
    )
    # Some legacy EMR HTML stores `&&nbsp;`; entity decoding leaves `&\xa0`.
    text = re.sub(r"&[\u00a0\u2007\u202f]+", " ", text)
    text = text.replace("\u00a0", " ").replace("\u2007", " ")
    text = text.replace("\u202f", " ").replace("\u3000", " ")
    return text


def contains_spacing_artifact(value: str) -> bool:
    return bool(HTML_SPACING_ARTIFACT_RE.search(str(value or "")))


def is_symbol_only_value(value: str) -> bool:
    text = clean_legacy_spacing_artifacts(value)
    return bool(text) and not MEANINGFUL_CONTENT_RE.search(text)


def has_abnormal_symbol_repetition(value: str, threshold: int = 8) -> bool:
    text = str(value or "")
    if len(re.findall(r"&[\u00a0\u2007\u202f]", text)) >= threshold:
        return True
    pattern = r"([^0-9A-Za-z\u4e00-\u9fff\s])(?:\s*\1){%d,}" % (
        threshold - 1
    )
    return bool(re.search(pattern, text))
