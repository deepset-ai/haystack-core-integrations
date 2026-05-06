# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import tempfile
from pathlib import Path
from typing import Any

from kreuzberg import (
    ExtractedImage,
    ExtractedTable,
    ExtractionConfig,
    ExtractionResult,
    config_to_json,
)


def _is_batch_error(result: ExtractionResult) -> bool:
    """
    Detect error results returned by kreuzberg's batch APIs.

    Batch APIs return ``ExtractionResult(content="Error: ...", metadata={},
    quality_score=None)`` instead of raising exceptions. Valid results always
    have populated metadata (at minimum ``output_format``).
    """
    return result.metadata == {} and result.quality_score is None


def _get_table_markdown(table: ExtractedTable | dict[str, Any]) -> str | None:
    """Get markdown string from a table (`ExtractedTable` object or dict)."""
    if isinstance(table, dict):
        return table.get("markdown") or None
    md = getattr(table, "markdown", None)
    return md or None


def _serialize_images(images: list[ExtractedImage]) -> list[dict[str, Any]]:
    """Serialize image metadata dicts, excluding binary data."""
    return [{k: v for k, v in img.items() if k != "data"} for img in images]


def _serialize_warnings(warnings: list[Any]) -> list[dict[str, str]]:
    """Serialize processing warnings to plain dicts."""
    serialized = []
    for w in warnings:
        if isinstance(w, dict):
            serialized.append({"source": w.get("source", ""), "message": w.get("message", "")})
        else:
            serialized.append({"source": getattr(w, "source", ""), "message": getattr(w, "message", "")})
    return serialized


def _serialize_keywords(keywords: list[Any]) -> list[dict[str, Any]]:
    """Serialize kreuzberg `ExtractedKeyword` objects to plain dicts (PyO3 objects aren't picklable)."""
    return [
        {
            "text": k.text,
            "score": k.score,
            "algorithm": k.algorithm,
            "positions": list(k.positions) if k.positions is not None else None,
        }
        for k in keywords
    ]


def _serialize_annotations(annotations: list[Any]) -> list[dict[str, Any]]:
    """Serialize PDF annotations to plain dicts."""
    serialized = []
    for ann in annotations:
        if isinstance(ann, dict):
            serialized.append(dict(ann))
        else:
            serialized.append(
                {
                    "type": getattr(ann, "annotation_type", None),
                    "content": getattr(ann, "content", None),
                    "page_number": getattr(ann, "page_number", None),
                }
            )
    return serialized


def _config_from_json_str(json_str: str) -> ExtractionConfig:
    """
    Load an `ExtractionConfig` from a JSON string via a temporary file.

    This is necessary because kreuzberg's PyO3 config objects don't expose a
    `from_json()` classmethod — only `from_file()`.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        tmp_path = f.name
        f.write(json_str)
    try:
        return ExtractionConfig.from_file(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _copy_config(config: ExtractionConfig) -> ExtractionConfig:
    """
    Deep copy an `ExtractionConfig` by round-tripping through JSON.

    This is necessary because kreuzberg's PyO3 config objects don't support
    Python's `copy.deepcopy()` protocol.
    """
    return _config_from_json_str(config_to_json(config))
