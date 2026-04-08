from __future__ import annotations

from collections import defaultdict, deque
from typing import TYPE_CHECKING, Optional

import numpy as np
from unstructured_inference.constants import Source
from unstructured_inference.inference.elements import TextRegion, TextRegions
from unstructured_inference.inference.layoutelement import (
    LayoutElement,
    LayoutElements,
    partition_groups_from_regions,
)

from unstructured.documents.elements import ElementType

if TYPE_CHECKING:
    from unstructured_inference.inference.elements import Rectangle


def build_text_region_from_coords(
    x1: int | float,
    y1: int | float,
    x2: int | float,
    y2: int | float,
    text: Optional[str] = None,
    source: Optional[Source] = None,
) -> TextRegion:
    """"""
    return TextRegion.from_coords(x1, y1, x2, y2, text=text, source=source)


def build_layout_element(
    bbox: "Rectangle",
    text: Optional[str] = None,
    source: Optional[Source] = None,
    element_type: Optional[str] = None,
) -> LayoutElement:
    """"""

    return LayoutElement(bbox=bbox, text=text, source=source, type=element_type)


def build_layout_elements_from_ocr_regions(
    ocr_regions: TextRegions,
    ocr_text: Optional[str] = None,
    group_by_ocr_text: bool = False,
) -> LayoutElements:
    """
    Get layout elements from OCR regions
    """

    grouped_regions = []
    if group_by_ocr_text:
        text_sections = ocr_text.split("\n\n")
        # Build mapping from text -> deque(indices) once (preserves OCR order)
        index_map: dict = defaultdict(deque)
        texts = ocr_regions.texts
        for idx, txt in enumerate(texts):
            index_map[txt].append(idx)

        for text_section in text_sections:
            regions = []
            words = text_section.replace("\n", " ").split()
            if not words:
                continue

            # For each word, take the earliest available OCR index (if any).
            # We pop from the deque to ensure indices are not reused across sections.
            remaining = len(words)
            for w in words:
                dq = index_map.get(w)
                if dq:
                    regions.append(dq.popleft())
                    remaining -= 1
                    if not dq:
                        # remove empty deque to keep map small
                        index_map.pop(w, None)
                    if remaining == 0:
                        break

            if not regions:
                continue

            # Ensure regions are in OCR order (original algorithm preserved OCR scan order)
            regions.sort()
            grouped_regions.append(ocr_regions.slice(regions))
    else:
        grouped_regions = partition_groups_from_regions(ocr_regions)

    merged_regions = TextRegions.from_list([merge_text_regions(group) for group in grouped_regions])
    return LayoutElements(
        element_coords=merged_regions.element_coords,
        texts=merged_regions.texts,
        sources=merged_regions.sources,
        element_class_ids=np.zeros(merged_regions.texts.shape),
        element_class_id_map={0: ElementType.UNCATEGORIZED_TEXT},
    )


def merge_text_regions(regions: TextRegions) -> TextRegion:
    """
    Merge a list of TextRegion objects into a single TextRegion.

    Parameters:
    - group (TextRegions): A group of TextRegion objects to be merged.

    Returns:
    - TextRegion: A single merged TextRegion object.
    """

    if not regions:
        raise ValueError("The text regions to be merged must be provided.")

    min_x1 = float(regions.x1.min())
    min_y1 = float(regions.y1.min())
    max_x2 = float(regions.x2.max())
    max_y2 = float(regions.y2.max())

    merged_text = " ".join(t for t in regions.texts if t)
    # assumption is the regions has the same source
    # assumption is the regions has the same source
    source = regions.sources[0]

    return TextRegion.from_coords(min_x1, min_y1, max_x2, max_y2, merged_text, source)
