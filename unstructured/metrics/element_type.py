from __future__ import annotations

import json

from typing_extensions import TypeAlias

FrequencyDict: TypeAlias = "dict[tuple[str, int | None], int]"
"""Like:
    {
        ("ListItem", 0): 2,
        ("NarrativeText", None): 2,
        ("Title", 0): 5,
        ("UncategorizedText", None): 6,
    }
"""


def get_element_type_frequency(
    elements: str,
) -> FrequencyDict:
    """
    Calculate the frequency of Element Types from a list of elements.

    Args:
        elements (str): String-formatted json of all elements (as a result of elements_to_json).
    Returns:
        Element type and its frequency in dictionary format.
    """
    frequency: dict[tuple[str, int | None], int] = {}
    if len(elements) == 0:
        return frequency
    for element in json.loads(elements):
        type = element.get("type")
        category_depth = element["metadata"].get("category_depth")
        key = (type, category_depth)
        if key not in frequency:
            frequency[key] = 1
        else:
            frequency[key] += 1
    return frequency


def calculate_element_type_percent_match(
    output: FrequencyDict,
    source: FrequencyDict,
    category_depth_weight: float = 0.5,
) -> float:
    """Calculate the percent match between two frequency dictionary.

    Intended to use with `get_element_type_frequency` function. The function counts the absolute
    exact match (type and depth), and counts the weighted match (correct type but different depth),
    then normalized with source's total elements.
    """
    if len(output) == 0 or len(source) == 0:
        return 0.0

    # Use the total of source values directly (equivalent to the original accumulation logic)
    total_source_element_count = sum(source.values())
    total_match_element_count = 0.0

    unmatched_depth_output: dict[str, int] = {}
    unmatched_depth_source: dict[str, int] = {}

    # Iterate output to compute exact matches and collect leftovers for both output and source
    for k, out_count in output.items():
        src_count = source.get(k, 0)
        if src_count:
            match_count = out_count if out_count <= src_count else src_count
            total_match_element_count += match_count

            leftover_output = out_count - match_count
            if leftover_output:
                element_type = k[0]
                if element_type not in unmatched_depth_output:
                    unmatched_depth_output[element_type] = leftover_output
                else:
                    unmatched_depth_output[element_type] += leftover_output

            leftover_source = src_count - match_count
            if leftover_source:
                element_type = k[0]
                if element_type not in unmatched_depth_source:
                    unmatched_depth_source[element_type] = leftover_source
                else:
                    unmatched_depth_source[element_type] += leftover_source
        else:
            # Key not present in source: all output counts are unmatched by depth
            element_type = k[0]
            if element_type not in unmatched_depth_output:
                unmatched_depth_output[element_type] = out_count
            else:
                unmatched_depth_output[element_type] += out_count

    # Add source-only keys (those not present in output) to unmatched_depth_source
    for k, src_count in source.items():
        if k in output:
            continue

        # add unmatched leftovers from output_copy to a new dictionary
        element_type = k[0]
        if element_type not in unmatched_depth_source:
            unmatched_depth_source[element_type] = src_count
        else:
            unmatched_depth_source[element_type] += src_count

    # Match any partially matched elements by type and apply the depth weight
    for element_type, src_count in unmatched_depth_source.items():
        out_count = unmatched_depth_output.get(element_type)
        if out_count:
            match_count = out_count if out_count <= src_count else src_count
            total_match_element_count += match_count * category_depth_weight

    return min(max(total_match_element_count / total_source_element_count, 0.0), 1.0)


def _convert_to_frequency_without_depth(d: FrequencyDict) -> dict[str, int]:
    """
    Takes in element frequency with depth of format (type, depth): value
    and converts to dictionary without depth of format type: value
    """
    res: dict[str, int] = {}
    for k, v in d.items():
        element_type = k[0]
        if element_type not in res:
            res[element_type] = v
        else:
            res[element_type] += v
    return res
