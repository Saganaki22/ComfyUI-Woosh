"""Woosh model loaders — consolidated."""

import logging
import os
import re
import folder_paths
import torch

from woosh.components.base import LoadConfig
from woosh.model.ldm import LatentDiffusionModel
from woosh.model.flowmap_from_pretrained import FlowMapFromPretrained
from woosh.model.video_kontext import VideoKontext

import comfy.model_management as mm

from ..types import GEN_MODEL
from .vram import WooshModelPatcher

log = logging.getLogger(__name__)

WOOSH_FOLDER = os.path.join(folder_paths.models_dir, "woosh")


def _device():
    return mm.get_torch_device()


def _woosh_path(model_name: str) -> str:
    return os.path.join(WOOSH_FOLDER, model_name)


_HIDDEN_FOLDERS = {"TextConditionerA", "TextConditionerV"}


def _get_model_names():
    if not os.path.isdir(WOOSH_FOLDER):
        return []
    names = []
    for entry in os.listdir(WOOSH_FOLDER):
        full = os.path.join(WOOSH_FOLDER, entry)
        if os.path.isdir(full) and os.path.isfile(os.path.join(full, "config.yaml")) and entry not in _HIDDEN_FOLDERS:
            names.append(entry)
    return sorted(names)


def _patch_config_paths_content(content: str) -> str:
    """Rewrite checkpoint paths in config YAML content to absolute paths.

    Matches both clean relative paths (checkpoints/Name) and stale absolute
    paths from a previous poisoned run (any .../models/woosh/Name).
    Always rewrites to the current WOOSH_FOLDER location.
    """
    woosh_url = WOOSH_FOLDER.replace("\\", "/")

    def _replace(m):
        name = m.group("name") or m.group("name2")
        return f"path: {woosh_url}/{name}"

    # Match: "path: checkpoints/Name" OR "path: <any>/models/woosh/Name"
    return re.sub(
        r"path:\s*checkpoints/(?P<name>\S+)|path:\s*\S*/models/woosh/(?P<name2>\S+)",
        _replace,
        content,
    )


def _patch_config_paths_temp(model_dir: str):
    """Temporarily patch config.yaml on disk for loading, return original content.

    The Woosh library reads config.yaml from disk when resolving sub-component
    paths (autoencoder, conditioners). We must patch the file so those resolve
    correctly, but we restore the original immediately after loading to prevent
    poisoning the config with machine-specific absolute paths.

    Returns the original file content (for _restore_config), or None if no
    patching was needed / the file didn't exist.
    """
    config_file = os.path.join(model_dir, "config.yaml")
    if not os.path.isfile(config_file):
        return None

    with open(config_file, "r", encoding="utf-8") as f:
        original = f.read()

    patched = _patch_config_paths_content(original)
    if patched == original:
        return None  # nothing to patch

    try:
        with open(config_file, "w", encoding="utf-8") as f:
            f.write(patched)
        return original
    except OSError:
        return None


def _restore_config(model_dir: str, original_content: str):
    """Restore original config.yaml after model loading."""
    if original_content is None:
        return
    config_file = os.path.join(model_dir, "config.yaml")
    try:
        with open(config_file, "w", encoding="utf-8") as f:
            f.write(original_content)
    except OSError:
        pass


# Model type to class mapping
GEN_MODEL_MAP = {
    "Flow": ("flow", LatentDiffusionModel),
    "DFlow": ("dflow", FlowMapFromPretrained),
    "VFlow": ("vflow", VideoKontext),
    "DVFlow": ("dvflow", FlowMapFromPretrained),
}


def _load_model(path: str, model_class):
    """Load a Woosh model with temporary config path patching.

    Patches config.yaml on disk, loads the model, then restores the original.
    The try/finally guarantees the original is always restored even on error.
    """
    original = _patch_config_paths_temp(path)
    try:
        model = model_class(LoadConfig(path=path))
        model = model.eval()
        _check_text_conditioner(model)
        return model
    finally:
        _restore_config(path, original)


def _check_text_conditioner(model):
    """Quick sanity check: text conditioner should produce non-trivial embeddings.

    If weights failed to load silently (strict=False fallback), the RoBERTa
    would produce near-zero or constant outputs, making the model ignore prompts.
    Only logs when something is wrong — silent on success.
    """
    try:
        raw = model
        if hasattr(raw, "conditioners"):
            cond = raw.conditioners
        else:
            return

        if "text" not in cond:
            return

        text_cond = cond["text"]
        device = next(text_cond.parameters()).device

        with torch.no_grad():
            batch = {
                "audio": torch.zeros(1, 1, 16000, device=device),
                "description": ["test sound effect"],
            }
            out = text_cond(batch, no_dropout=True, no_cond=False, device=device)
            emb = out["text_cond"]

            emb_norm = emb.norm().item()
            emb_std = emb.std().item()

            if emb_norm < 1e-6:
                log.error(
                    "[Woosh] Text conditioner producing near-ZERO embeddings! "
                    "Weights likely failed to load. Prompts will be ignored. "
                    "Check that TextConditionerA/weights.safetensors is valid."
                )
            elif emb_std < 1e-8:
                log.error(
                    "[Woosh] Text conditioner producing CONSTANT embeddings! "
                    "Weights likely failed to load. Prompts will be ignored. "
                    "Check that TextConditionerA/weights.safetensors is valid."
                )
    except Exception:
        pass


class WooshLoadFlow:
    """Unified loader for all generative models: Flow, DFlow, VFlow, DVFlow."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (_get_model_names(), {"tooltip": "Select model checkpoint folder"}),
                "model_type": (list(GEN_MODEL_MAP.keys()), {"tooltip": "Flow = full ODE sampler (best quality, 50 steps). DFlow = distilled (fast, 4 steps). VFlow = video-to-audio full. DVFlow = video-to-audio distilled. model_name must match — e.g. Woosh-Flow for Flow, Woosh-DFlow for DFlow"}),
            }
        }

    RETURN_TYPES = (GEN_MODEL,)
    RETURN_NAMES = ("gen_model",)
    FUNCTION = "load"
    CATEGORY = "Woosh/Loaders"
    DESCRIPTION = "Load Woosh generative model (Flow, DFlow, VFlow, or DVFlow)"

    def __init__(self):
        self._model = None
        self._key = None

    def load(self, model_name, model_type):
        key = (model_name, model_type)
        if self._model is not None and self._key == key:
            return (self._model,)

        # Evict previous model if type changed
        if self._model is not None:
            self._model.force_unload()

        path = _woosh_path(model_name)
        _, model_class = GEN_MODEL_MAP[model_type]
        model = _load_model(path, model_class)

        def _reload():
            return _load_model(path, model_class)

        self._model = WooshModelPatcher(model, vram_size_gb=4.0, reload_fn=_reload)
        self._key = key
        return (self._model,)


NODE_CLASS_MAPPINGS_LOADERS = {
    "WooshLoadFlow": WooshLoadFlow,
}

NODE_DISPLAY_MAPPINGS_LOADERS = {
    "WooshLoadFlow": "Woosh Model Loader",
}
