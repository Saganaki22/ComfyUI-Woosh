"""Woosh model loaders — consolidated."""

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


def _patch_config_paths(model_dir: str):
    """Rewrite checkpoints/XXX paths in config.yaml to absolute model paths.
    Woosh library resolves paths relative to CWD, but ComfyUI stores models
    in models/woosh/. This patches config.yaml so nested component references
    (autoencoder, conditioners) resolve correctly.
    """
    config_file = os.path.join(model_dir, "config.yaml")
    if not os.path.isfile(config_file):
        return

    with open(config_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Replace checkpoints/ModelName with absolute path
    def _replace(m):
        name = m.group(1)
        return os.path.join(WOOSH_FOLDER, name).replace("\\", "/")

    patched = re.sub(r"path:\s*checkpoints/(\S+)", lambda m: f"path: {_replace(m)}", content)

    if patched != content:
        with open(config_file, "w", encoding="utf-8") as f:
            f.write(patched)


# Model type to class mapping
GEN_MODEL_MAP = {
    "Flow": ("flow", LatentDiffusionModel),
    "DFlow": ("dflow", FlowMapFromPretrained),
    "VFlow": ("vflow", VideoKontext),
    "DVFlow": ("dvflow", FlowMapFromPretrained),
}


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
        _patch_config_paths(path)
        _, model_class = GEN_MODEL_MAP[model_type]
        model = model_class(LoadConfig(path=path))
        model = model.eval()

        def _reload():
            _patch_config_paths(path)
            m = model_class(LoadConfig(path=path))
            return m.eval()

        self._model = WooshModelPatcher(model, vram_size_gb=4.0, reload_fn=_reload)
        self._key = key
        return (self._model,)


NODE_CLASS_MAPPINGS_LOADERS = {
    "WooshLoadFlow": WooshLoadFlow,
}

NODE_DISPLAY_MAPPINGS_LOADERS = {
    "WooshLoadFlow": "Woosh Model Loader",
}
