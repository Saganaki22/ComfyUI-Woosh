"""Woosh ComfyUI Custom Nodes — Sound Effect Foundation Model by Sony AI."""

import logging
import os
import sys

# Add bundled Woosh package to Python path (no pip install needed)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Woosh"))

import folder_paths
from .nodes.model_paths import (
    DEFAULT_MMAUDIO_FOLDER,
    DEFAULT_WOOSH_FOLDER,
    list_woosh_model_names,
)

# Register model folder so ComfyUI finds Woosh checkpoints
WOOSH_FOLDER = DEFAULT_WOOSH_FOLDER
MMAUDIO_FOLDER = DEFAULT_MMAUDIO_FOLDER
folder_paths.add_model_folder_path("woosh", WOOSH_FOLDER)
folder_paths.add_model_folder_path("mmaudio", MMAUDIO_FOLDER)

# Set HF cache BEFORE transformers is imported by anything.
# Must happen before the Woosh library or any other node triggers
# `import transformers` — otherwise HF_HOME is baked to a wrong default.
_HF_CACHE = os.path.join(WOOSH_FOLDER, "hf_cache")
_HF_HUB_CACHE = os.path.join(_HF_CACHE, "hub")
os.makedirs(_HF_CACHE, exist_ok=True)
os.environ.setdefault("HF_HOME", _HF_CACHE)
os.environ.setdefault("TRANSFORMERS_CACHE", _HF_HUB_CACHE)
os.environ.setdefault("HF_HUB_CACHE", _HF_HUB_CACHE)

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

    cache_dir = _HF_HUB_CACHE if os.path.isdir(_HF_HUB_CACHE) else _HF_CACHE

    def _kwargs(kwargs, *, local_files_only=None, force_download=None):
        patched = dict(kwargs)
        patched.setdefault("cache_dir", cache_dir)
        if local_files_only is not None:
            patched["local_files_only"] = local_files_only
        if force_download is not None:
            patched["force_download"] = force_download
        return patched

    def _tokenizer_ok(tokenizer):
        return getattr(tokenizer, "vocab_size", 0) > 1000

    def _config_ok(config):
        return getattr(config, "hidden_size", 0) > 100

    def _tok(cls, *args, **kwargs):
        try:
            tokenizer = _orig_tok(
                cls, *args, **_kwargs(kwargs, local_files_only=True)
            )
            if _tokenizer_ok(tokenizer):
                return tokenizer
        except Exception:
            pass

        tokenizer = _orig_tok(cls, *args, **_kwargs(kwargs, local_files_only=False))
        if not _tokenizer_ok(tokenizer):
            tokenizer = _orig_tok(
                cls,
                *args,
                **_kwargs(kwargs, local_files_only=False, force_download=True),
            )
        if not _tokenizer_ok(tokenizer):
            raise RuntimeError(
                f"Loaded an invalid RoBERTa tokenizer from {cache_dir}. "
                "Delete the roberta-large cache folder and let Woosh download it again."
            )
        return tokenizer

    def _cfg(cls, *args, **kwargs):
        try:
            config = _orig_cfg(cls, *args, **_kwargs(kwargs, local_files_only=True))
            if _config_ok(config):
                return config
        except Exception:
            pass

        config = _orig_cfg(cls, *args, **_kwargs(kwargs, local_files_only=False))
        if not _config_ok(config):
            config = _orig_cfg(
                cls,
                *args,
                **_kwargs(kwargs, local_files_only=False, force_download=True),
            )
        return config

    def _mdl(cls, *args, **kwargs):
        try:
            return _orig_model(cls, *args, **_kwargs(kwargs, local_files_only=True))
        except Exception:
            return _orig_model(cls, *args, **_kwargs(kwargs, local_files_only=False))

    RobertaTokenizer.from_pretrained = classmethod(_tok)
    AutoConfig.from_pretrained = classmethod(_cfg)
    RobertaModel.from_pretrained = classmethod(_mdl)


_patch_hf_offline()

def get_woosh_model_names():
    """List Woosh checkpoint folders from every registered ComfyUI model path."""
    return list_woosh_model_names()


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
