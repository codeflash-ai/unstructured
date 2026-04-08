from typing import Any, Dict, List, Optional

from unstructured.documents.elements import Text

_DATASAUR_REQUIRED_FIELDS = (
    ("text", str),
    ("type", str),
    ("start_idx", int),
    ("end_idx", int),
)


def stage_for_datasaur(
    elements: List[Text],
    entities: Optional[List[List[Dict[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    """Convert a list of elements into a list of dictionaries for use in Datasaur"""
    _entities: List[List[Dict[str, Any]]] = [[] for _ in range(len(elements))]
    if entities is not None:
        if len(entities) != len(elements):
            raise ValueError("If entities is specified, it must be the same length as elements.")

        for entity_list in entities:
            for entity in entity_list:
                _validate_datasaur_entity(entity)

        _entities = entities

    result = [{"text": item.text, "entities": _entities[i]} for i, item in enumerate(elements)]

    return result


def _validate_datasaur_entity(entity: Dict[str, Any]):
    """Raises an error if the Datasaur entity is invalid."""
    for key, _type in _DATASAUR_REQUIRED_FIELDS:
        try:
            value = entity[key]
        except KeyError:
            raise ValueError(f"Key '{key}' was expected but not present in the Datasaur entity.")

        if not isinstance(value, _type):
            raise ValueError(f"Expected type {_type} for {key}. Got {type(key)}.")
