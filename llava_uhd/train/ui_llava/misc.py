import json

def is_serializable(obj):
    try:
        json.dumps(obj)
        return True
    except (TypeError, OverflowError):
        return False

def clean_dict(data):
    if isinstance(data, dict):
        return {k: clean_dict(v) for k, v in data.items() if is_serializable(v)}
    elif isinstance(data, list):
        return [clean_dict(item) for item in data if is_serializable(item)]
    else:
        return data if is_serializable(data) else None

def remove_unserializable_items(data):
    """
    Recursively remove all unserializable items from dictionaries or lists.

    Args:
    data (dict or list): The dictionary or list from which to remove unserializable items.

    Returns:
    dict or list: A cleaned version of the input with all unserializable items removed.
    """
    return clean_dict(data)