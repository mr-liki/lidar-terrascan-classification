"""Class codes: the engine's mapping JSON and the codes the post-processing works with.

``class_mapping`` maps the model's internal classes 0-4 (ground, low / medium / high vegetation,
building) to delivered codes, ``power_mapping`` the power-line classes 5 (wire) and 6 (pole).
Naksha supplies both from its dialog; the standalone app resolves them from the .ptc file.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

from .ptc import resolve_ptc

CLASS_KEYS = {0: "ground", 1: "low_veg", 2: "medium_veg", 3: "high_veg", 4: "building"}
WIRE_INTERNAL_KEY = 5
POLE_INTERNAL_KEY = 6

# Host no-PTC convention: LAS 0 unclassified, 7 low point, 18 high noise.
DEFAULT_UNCLASSIFIED = 0
DEFAULT_LOW_NOISE = 7
DEFAULT_HIGH_NOISE = 18

# ASPRS codes, used for a class the .ptc does not name.
DEFAULT_CLASS_MAPPING = {0: 2, 1: 3, 2: 4, 3: 5, 4: 6}
DEFAULT_POWER_MAPPING = {WIRE_INTERNAL_KEY: 14, POLE_INTERNAL_KEY: 15}

_SEMANTIC_BY_INTERNAL = {0: "ground", 1: "low_vegetation", 2: "medium_vegetation",
                         3: "high_vegetation", 4: "building"}


def _next_free(preferred: int, used: set) -> int:
    code = int(preferred)
    while code in used and code < 255:
        code += 1
    return code


def build_engine_mapping(class_mapping: dict, power_mapping: dict, advanced_config: dict,
                         enable_power_lines: bool) -> dict:
    """The mapping JSON the engine reads, from the supplied class codes.

    Unclassified / low point / high noise come from the active PTC when it defines them, else
    the host convention (0 / 7 / 18). A support code that would collide with a class code moves
    to the next free number, so a point can never be delivered under a class it does not belong
    to.
    """
    classes: Dict[str, int] = {}
    for index, key in CLASS_KEYS.items():
        if index not in class_mapping:
            raise ValueError(f"The class mapping has no output code for '{key}'.")
        classes[key] = int(class_mapping[index])
    classes["wire"] = int((power_mapping or {}).get(WIRE_INTERNAL_KEY, 14))
    classes["pole"] = int((power_mapping or {}).get(POLE_INTERNAL_KEY, 15))

    used = set(classes.values()) if enable_power_lines else {classes[k] for k in CLASS_KEYS.values()}
    optional = dict((advanced_config or {}).get("_ptc_optional_semantic_codes", {}) or {})
    unclassified = _next_free(optional.get("uncategorized", DEFAULT_UNCLASSIFIED), used)
    # support classes may share a code with each other (a PTC often has one noise class)
    low_noise = _next_free(optional.get("low_point", DEFAULT_LOW_NOISE), used | {unclassified})
    high_noise = _next_free(optional.get("high_noise", DEFAULT_HIGH_NOISE), used | {unclassified})
    classes.update({"unclassified": unclassified, "low_noise": low_noise, "high_noise": high_noise})
    return {"classes": classes, "power_lines_enabled": bool(enable_power_lines)}


def mappings_from_ptc(ptc_path) -> Tuple[dict, dict, dict, dict]:
    """(class_mapping, power_mapping, optional_codes, notes) resolved from a .ptc file.

    ``optional_codes`` is the ``_ptc_optional_semantic_codes`` entry of the advanced config;
    ``notes`` maps each class to where its code came from.
    """
    resolution = resolve_ptc(ptc_path)
    found = resolution.mapping
    notes: Dict[str, str] = {}
    class_mapping = {}
    for internal, semantic in _SEMANTIC_BY_INTERNAL.items():
        if semantic in found:
            class_mapping[internal] = int(found[semantic])
            notes[semantic] = f"PTC code {found[semantic]}"
        else:
            class_mapping[internal] = DEFAULT_CLASS_MAPPING[internal]
            notes[semantic] = f"not in the PTC; ASPRS code {DEFAULT_CLASS_MAPPING[internal]}"
    power_mapping = {}
    for internal, semantic in ((WIRE_INTERNAL_KEY, "wire"), (POLE_INTERNAL_KEY, "pole")):
        if semantic in found:
            power_mapping[internal] = int(found[semantic])
            notes[semantic] = f"PTC code {found[semantic]}"
        else:
            power_mapping[internal] = DEFAULT_POWER_MAPPING[internal]
            reason = ("ambiguous in the PTC" if semantic in resolution.ambiguous
                      else "not in the PTC")
            notes[semantic] = f"{reason}; code {DEFAULT_POWER_MAPPING[internal]}"
    optional = {key: int(found[key]) for key in ("uncategorized", "low_point", "high_noise")
                if key in found}
    return class_mapping, power_mapping, optional, notes


def engine_classes(class_mapping: dict, power_mapping: dict, advanced_config: Optional[dict],
                   enable_power_lines: bool) -> Dict[str, int]:
    return build_engine_mapping(class_mapping, power_mapping, advanced_config or {},
                                enable_power_lines)["classes"]
