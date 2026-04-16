"""Woosh ComfyUI Custom Nodes — Sound Effect Foundation Model by Sony AI."""

import logging
import os
import sys

# Add bundled Woosh package to Python path (no pip install needed)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Woosh"))

import folder_paths

# Register model folder so ComfyUI finds Woosh checkpoints
WOOSH_FOLDER = os.path.join(folder_paths.models_dir, "woosh")
folder_paths.add_model_folder_path("woosh", WOOSH_FOLDER)

# Set HF cache BEFORE transformers is imported by anything.
# Must happen before the Woosh library or any other node triggers
# `import transformers` — otherwise HF_HOME is baked to a wrong default.
_HF_CACHE = os.path.join(WOOSH_FOLDER, "hf_cache")
os.makedirs(_HF_CACHE, exist_ok=True)
os.environ.setdefault("HF_HOME", _HF_CACHE)
os.environ.setdefault("TRANSFORMERS_CACHE", _HF_CACHE)
os.environ.setdefault("HF_HUB_CACHE", os.path.join(_HF_CACHE, "hub"))

# Suppress Woosh library INFO logs (Loading config, Loading weights, etc.)
logging.getLogger("woosh").setLevel(logging.WARNING)


# Monkey-patch HuggingFace from_pretrained to try offline first, download only if missing.
# The Woosh library loads RoBERTa tokenizer inside model __init__ — we can't control that,
# so we patch at the transformers level.
def _patch_hf_offline():
    try:
        from transformers import RobertaTokenizer, AutoConfig, RobertaModel
    except ImportError:
        return

    _orig_tok = RobertaTokenizer.from_pretrained.__func__
    _orig_cfg = AutoConfig.from_pretrained.__func__
    _orig_model = RobertaModel.from_pretrained.__func__

    def _tok(cls, *args, **kwargs):
        kwargs.setdefault("cache_dir", _HF_CACHE)
        try:
            return _orig_tok(cls, *args, local_files_only=True, **kwargs)
        except Exception:
            return _orig_tok(cls, *args, **kwargs)

    def _cfg(cls, *args, **kwargs):
        kwargs.setdefault("cache_dir", _HF_CACHE)
        try:
            return _orig_cfg(cls, *args, local_files_only=True, **kwargs)
        except Exception:
            return _orig_cfg(cls, *args, **kwargs)

    def _mdl(cls, *args, **kwargs):
        kwargs.setdefault("cache_dir", _HF_CACHE)
        try:
            return _orig_model(cls, *args, local_files_only=True, **kwargs)
        except Exception:
            return _orig_model(cls, *args, **kwargs)

    RobertaTokenizer.from_pretrained = classmethod(_tok)
    AutoConfig.from_pretrained = classmethod(_cfg)
    RobertaModel.from_pretrained = classmethod(_mdl)


_patch_hf_offline()

_HIDDEN_FOLDERS = {"TextConditionerA", "TextConditionerV"}


def get_woosh_model_names():
    """List subdirectories of models/woosh/ — each is a model checkpoint folder."""
    if not os.path.isdir(WOOSH_FOLDER):
        return []
    names = []
    for entry in os.listdir(WOOSH_FOLDER):
        full = os.path.join(WOOSH_FOLDER, entry)
        if (
            os.path.isdir(full)
            and os.path.isfile(os.path.join(full, "config.yaml"))
            and entry not in _HIDDEN_FOLDERS
        ):
            names.append(entry)
    return sorted(names)


# Import all node mappings
from .nodes.loaders import (
    NODE_CLASS_MAPPINGS_LOADERS,
    NODE_DISPLAY_MAPPINGS_LOADERS,
)
from .nodes.samplers import (
    NODE_CLASS_MAPPINGS_SAMPLERS,
    NODE_DISPLAY_MAPPINGS_SAMPLERS,
)
from .nodes.video import (
    NODE_CLASS_MAPPINGS_VIDEO,
    NODE_DISPLAY_MAPPINGS_VIDEO,
)
from .nodes.utils import (
    NODE_CLASS_MAPPINGS_UTILS,
    NODE_DISPLAY_MAPPINGS_UTILS,
)

NODE_CLASS_MAPPINGS = {
    **NODE_CLASS_MAPPINGS_LOADERS,
    **NODE_CLASS_MAPPINGS_SAMPLERS,
    **NODE_CLASS_MAPPINGS_VIDEO,
    **NODE_CLASS_MAPPINGS_UTILS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **NODE_DISPLAY_MAPPINGS_LOADERS,
    **NODE_DISPLAY_MAPPINGS_SAMPLERS,
    **NODE_DISPLAY_MAPPINGS_VIDEO,
    **NODE_DISPLAY_MAPPINGS_UTILS,
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "get_woosh_model_names"]
