import importlib


def import_model_class_by_version(version: str):
    assert version in ['v2', 'v3'], f'Unsupported model version: {version}'
    return getattr(importlib.import_module(f'.{version}', __package__), 'MoGeModel')
