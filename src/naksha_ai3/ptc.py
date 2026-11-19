"""Read a .ptc class-definition file and resolve its semantic classes (Ground, Low Vegetation...).

Standard library only. Kept in step with Naksha's gui/AI/common/ptc_semantics.py (the GUI-state
parts are omitted: the standalone app reads the .ptc file itself).
"""
from __future__ import annotations

import re
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple


BASE_INTERNAL_TO_SEMANTIC = {
    0: "ground",
    1: "low_vegetation",
    2: "medium_vegetation",
    3: "high_vegetation",
    4: "building",
}


POWER_INTERNAL_TO_SEMANTIC = {
    5: "wire",
    6: "pole",
}

SEMANTIC_LABELS = {
    "ground": "Ground",
    "low_vegetation": "Low Vegetation",
    "medium_vegetation": "Medium Vegetation",
    "high_vegetation": "High Vegetation",
    "building": "Building",
    "low_point": "Low Point",
    "high_noise": "High Noise",
    "uncategorized": "Uncategorized",
    "wire": "Power Line Wire",
    "pole": "Power Line Pole/Pylon",
}

# Aliases are deliberately conservative.  The semantic resolver should not
# guess a class just because a vague word occurs in a PTC description.
_ALIASES = {
    "ground": (
        "ground",
        "terrain",
    ),
    "low_vegetation": (
        "low vegetation",
        "low veg",
        "lowveg",
    ),
    "medium_vegetation": (
        "medium vegetation",
        "medium veg",
        "mediumveg",
        "mid vegetation",
        "mid veg",
        "midveg",
    ),
    "high_vegetation": (
        "high vegetation",
        "high veg",
        "highveg",
    ),
    "building": (
        "building",
        "buildings",
    ),
    "low_point": (
        "low point",
        "low points",
        "lowpoint",
        "lowpoints",
    ),
    "high_noise": (
        "high noise",
        "highnoise",
        "high point noise",
        "high points noise",
        "high outlier",
        "high outliers",
    ),
    "uncategorized": (
        "uncategorized",
        "uncategorised",
        "unclassified",
        "unknown",
        "not classified",
        "unclassified point",
        "unclassified points",
        "created never classified",
        "never classified",
        "not categorized",
        "not categorised",
    ),
    "wire": (
        "bare conductors",
        "bare conductor",
        "power line wire",
        "powerline wire",
        "power wire",
        "conductor",
        "conductors",
    ),
    "pole": (
        "pylons or poles",
        "pylon or pole",
        "power line pole",
        "powerline pole",
        "power pole",
        "pylon",
        "pylons",
        "pole",
        "poles",
        "tower",
        "towers",
    ),
}


def _normalize(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("_", " ").replace("-", " ").replace("/", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _alias_score(field_text: str, alias: str) -> int:
    field = _normalize(field_text)
    target = _normalize(alias)
    if not field or not target:
        return 0
    if field == target:
        return 1000 + len(target)
    # Whole-phrase containment is weaker than an exact match but still valid.
    if re.search(r"(?:^|\s)" + re.escape(target) + r"(?:$|\s)", field):
        return 500 + len(target)
    return 0


def _entry_score(info: Mapping[str, Any], semantic: str) -> int:
    aliases = _ALIASES.get(semantic, ())
    best = 0
    # lvl is frequently the real semantic label in Nakshatech PTC files, while
    # description can be blank or project-specific.  Score both independently.
    for field_name in ("lvl", "description", "name", "label"):
        field = info.get(field_name, "")
        for alias in aliases:
            best = max(best, _alias_score(field, alias))
    return best


def _coerce_palette(palette: Any) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    if not isinstance(palette, Mapping):
        return out
    for raw_code, raw_info in palette.items():
        try:
            code = int(raw_code)
        except Exception:
            continue
        if not 0 <= code <= 255:
            continue
        info = dict(raw_info) if isinstance(raw_info, Mapping) else {}
        out[code] = info
    return out


@dataclass(frozen=True)
class SemanticResolution:
    mapping: Dict[str, int]
    source: str
    ptc_path: str
    matched: Dict[str, str]
    ambiguous: Dict[str, Tuple[int, ...]]

    def code(self, semantic: str) -> Optional[int]:
        value = self.mapping.get(str(semantic))
        return None if value is None else int(value)


def resolve_semantic_mapping(
    palette: Any,
    *,
    class_mapping: Optional[Mapping[Any, Any]] = None,
    power_mapping: Optional[Mapping[Any, Any]] = None,
    source: str = "palette",
    ptc_path: str = "",
) -> SemanticResolution:
    """Resolve semantic names to numeric class codes from an active PTC palette.

    Base and power mappings supplied by the AI dialog are used only as explicit
    fallbacks.  Low Point and Uncategorized are never invented from arbitrary
    codes: they must exist semantically in the PTC, except that code 0 may be
    used as an Uncategorized fallback when the active palette actually contains
    code 0 and no other semantic has claimed it.
    """
    pal = _coerce_palette(palette)
    mapping: Dict[str, int] = {}
    matched: Dict[str, str] = {}
    ambiguous: Dict[str, Tuple[int, ...]] = {}

    for semantic in SEMANTIC_LABELS:
        scored = []
        for code, info in pal.items():
            score = _entry_score(info, semantic)
            if score > 0:
                scored.append((score, code, info))
        if not scored:
            continue
        scored.sort(key=lambda row: (-row[0], row[1]))
        best_score = scored[0][0]
        best_rows = [row for row in scored if row[0] == best_score]
        if len(best_rows) > 1:
            # Production rule: an ambiguous PTC label must never silently pick
            # one numeric code. Base/power classes can still fall back to the
            # explicit dialog mapping; Low Point/Uncategorized are skipped.
            ambiguous[semantic] = tuple(int(row[1]) for row in best_rows)
            continue
        chosen = best_rows[0]
        mapping[semantic] = int(chosen[1])
        info = chosen[2]
        matched[semantic] = (
            f"code={chosen[1]} description={info.get('description','')!r} "
            f"lvl={info.get('lvl','')!r} score={best_score}"
        )

    # Explicit dialog mapping is authoritative for the five trained classes if
    # a PTC label was absent/atypical.  This preserves compatibility with the
    # existing Basic/Advanced/Premium routing without hard-coding LAS numbers.
    if class_mapping:
        for internal, semantic in BASE_INTERNAL_TO_SEMANTIC.items():
            if semantic in mapping:
                continue
            try:
                value = int(class_mapping[internal])
            except Exception:
                continue
            if 0 <= value <= 255:
                mapping[semantic] = value
                matched[semantic] = f"fallback: class_mapping[{internal}]={value}"

    if power_mapping:
        for internal, semantic in POWER_INTERNAL_TO_SEMANTIC.items():
            if semantic in mapping:
                continue
            try:
                value = int(power_mapping[internal])
            except Exception:
                continue
            if 0 <= value <= 255:
                mapping[semantic] = value
                matched[semantic] = f"fallback: power_mapping[{internal}]={value}"

    # A large number of Nakshatech PTCs use class 0 as Unclassified but leave
    # the text field blank.  Falling back to 0 is safe only when code 0 is
    # actually present in the active PTC and is not already assigned to another
    # known semantic.
    if "uncategorized" not in mapping and 0 in pal and 0 not in mapping.values():
        mapping["uncategorized"] = 0
        matched["uncategorized"] = "fallback: active PTC contains unclaimed code 0"

    return SemanticResolution(
        mapping=mapping,
        source=str(source),
        ptc_path=str(ptc_path or ""),
        matched=matched,
        ambiguous=ambiguous,
    )


def _parse_ptc_palette(path_value: Any) -> Dict[int, Dict[str, Any]]:
    """Read the active Nakshatech-style PTC without depending on GUI state.

    Display/session restores can preserve color/show/weight while accidentally
    dropping ``description``/``lvl``.  The .ptc file is the authoritative
    semantic source, so V3.23 reads its two-line records directly when possible.
    """
    text_path = str(path_value or "").strip()
    if not text_path:
        return {}
    try:
        path = Path(text_path)
        if not path.is_file():
            return {}
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            lines = [ln.strip() for ln in fh if ln.strip()]
    except Exception:
        return {}

    out: Dict[int, Dict[str, Any]] = {}
    for i in range(0, max(0, len(lines) - 1), 2):
        try:
            header = lines[i].split("\t")
            detail = lines[i + 1].split("\t")
            if len(header) < 2:
                continue
            code = int(header[0])
            if not 0 <= code <= 255:
                continue
            desc = header[1] if len(header) > 1 else ""
            lvl = header[2] if len(header) > 2 else ""
            draw = detail[1] if len(detail) > 1 else ""
            out[code] = {
                "description": str(desc),
                "lvl": str(lvl),
                "draw": str(draw),
            }
        except Exception:
            continue
    return out


def format_resolution(resolution: SemanticResolution) -> str:
    lines = [
        "ACTIVE PTC SEMANTIC SNAPSHOT",
        f"  source: {resolution.source}",
    ]
    if resolution.ptc_path:
        lines.append(f"  ptc: {resolution.ptc_path}")
    for semantic, label in SEMANTIC_LABELS.items():
        code = resolution.mapping.get(semantic)
        if code is not None:
            lines.append(f"  {label:<24} -> class {int(code)}")
    if resolution.ambiguous:
        lines.append(f"  ambiguous: {resolution.ambiguous}")
    return "\n".join(lines)


def parse_ptc(path) -> Dict[int, Dict[str, Any]]:
    """Classes of a .ptc file: {code: {description, lvl, draw}}."""
    return _parse_ptc_palette(path)


def resolve_ptc(path) -> SemanticResolution:
    """Semantic class codes of a .ptc file."""
    palette = _parse_ptc_palette(path)
    if not palette:
        raise ValueError(f"Not a readable .ptc class-definition file: {path}")
    return resolve_semantic_mapping(palette, source="ptc_file", ptc_path=str(path))
