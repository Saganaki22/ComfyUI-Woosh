"""Woosh ComfyUI Custom Nodes — Sound Effect Foundation Model by Sony AI."""

import logging
import os
import sys
import folder_paths

# Add bundled Woosh package to Python path (no pip install needed)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Woosh"))

# Suppress Woosh library INFO logs (Loading config, Loading weights, etc.)
# Only warnings and errors will show in console
logging.getLogger("woosh").setLevel(logging.WARNING)

# Register model folder so ComfyUI finds Woosh checkpoints
WOOSH_FOLDER = os.path.join(folder_paths.models_dir, "woosh")
folder_paths.add_model_folder_path("woosh", WOOSH_FOLDER)

# Cache HuggingFace downloads (RoBERTa tokenizer) locally in woosh model folder
_HF_CACHE = os.path.join(WOOSH_FOLDER, "hf_cache")
os.makedirs(_HF_CACHE, exist_ok=True)
os.environ["HF_HOME"] = _HF_CACHE
os.environ["TRANSFORMERS_CACHE"] = _HF_CACHE
os.environ["HF_HUB_CACHE"] = os.path.join(_HF_CACHE, "hub")

# Monkey-patch HuggingFace from_pretrained to try offline first, download only if missing.
# The Woosh library loads RoBERTa tokenizer inside model __init__ — we can't control that,
# so we patch at the transformers level.
def _patch_hf_offline():
    try:
        from transformers import RobertaTokenizer, AutoConfig, RobertaModel
    except ImportError:
        return

    # Get the underlying functions (bypass classmethod binding)
    _orig_tok = RobertaTokenizer.from_pretrained.__func__
    _orig_cfg = AutoConfig.from_pretrained.__func__
    _orig_model = RobertaModel.from_pretrained.__func__

    def _tok(cls, *args, **kwargs):
        try:
            return _orig_tok(cls, *args, local_files_only=True, **kwargs)
        except Exception:
            return _orig_tok(cls, *args, **kwargs)

    def _cfg(cls, *args, **kwargs):
        try:
            return _orig_cfg(cls, *args, local_files_only=True, **kwargs)
        except Exception:
            return _orig_cfg(cls, *args, **kwargs)

    def _mdl(cls, *args, **kwargs):
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
        if os.path.isdir(full) and os.path.isfile(os.path.join(full, "config.yaml")) and entry not in _HIDDEN_FOLDERS:
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
