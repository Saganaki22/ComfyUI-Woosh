"""Woosh samplers — consolidated text encode + single unified sampler."""

import os
import sys
import json
import logging
import subprocess
import tempfile
import gc
import torch
import folder_paths
import comfy.model_management as mm
from comfy.model_patcher import ModelPatcher

from woosh.components.base import LoadConfig
from woosh.components.clap_conditioners import SFXCLAPTextConditioner
from woosh.inference.flowmap_sampler import sample_euler
from woosh.model.flowmap_from_pretrained import FlowMapFromPretrained
from woosh.model.video_kontext import VideoKontext
from woosh.utils.video import SynchformerProcessor

from ..woosh_types import GEN_MODEL, TEXT_COND, VIDEO, AUDIO
from .model_paths import (
    DEFAULT_WOOSH_FOLDER,
    get_mmaudio_folders,
    get_woosh_folders,
    resolve_woosh_path,
)

log = logging.getLogger(__name__)
SAMPLE_RATE = 48000
LATENT_CHANNELS = 128

WOOSH_FOLDER = DEFAULT_WOOSH_FOLDER
_WORKER_SCRIPT = os.path.join(os.path.dirname(__file__), "_infer_worker.py")


def _clear_torch_memory():
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mps = getattr(torch, "mps", None)
    if mps is not None and hasattr(mps, "empty_cache"):
        try:
            mps.empty_cache()
        except RuntimeError:
            pass
    mm.soft_empty_cache()


def _device():
    return mm.get_torch_device()


def _offload_device():
    return mm.intermediate_device()


def _seed_noise(seed, shape, device):
    torch.manual_seed(seed)
    return torch.randn(shape, device=device, dtype=torch.float32)


def _flowmatching_solver_dtype(device):
    device_type = getattr(device, "type", str(device))
    return torch.float32 if device_type == "mps" else torch.float64


def _normalize_audio(audio):
    peak = audio.abs().amax(dim=-1, keepdim=True)
    peak = torch.clamp(peak, min=1.0)
    return audio / peak


def _woosh_path(name: str) -> str:
    return resolve_woosh_path(name)


def _subprocess_infer(
    model_dir,
    prompt,
    seed,
    cfg,
    latent_frames,
    steps,
    model_type,
    video=None,
    text_conditioner_dir=None,
):
    woosh_pkg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Woosh")
    hf_cache = os.path.join(WOOSH_FOLDER, "hf_cache")
    output_path = os.path.join(
        tempfile.gettempdir(), f"woosh_out_{os.getpid()}_{seed}.pt"
    )

    video_path = None
    video_fps = None
    cleanup_video = None

    if video is not None:
        video_path = os.path.join(
            tempfile.gettempdir(), f"woosh_vid_{os.getpid()}_{seed}.pt"
        )
        torch.save(video["frames"].cpu(), video_path)
        video_fps = video["rate"]
        cleanup_video = video_path

    payload = json.dumps(
        {
            "model_dir": model_dir,
            "prompt": prompt,
            "seed": seed,
            "cfg": cfg,
            "latent_frames": latent_frames,
            "steps": steps,
            "is_distilled": model_type in ("dflow", "dvflow"),
            "model_type": model_type,
            "output_path": output_path,
            "woosh_pkg_path": woosh_pkg_path,
            "hf_cache": hf_cache,
            "woosh_folder": WOOSH_FOLDER,
            "woosh_folders": get_woosh_folders(),
            "mmaudio_folders": get_mmaudio_folders(),
            "models_dir": folder_paths.models_dir,
            "video_path": video_path,
            "video_fps": video_fps,
            "text_conditioner_dir": text_conditioner_dir,
        }
    )

    proc = subprocess.run(
        [sys.executable, _WORKER_SCRIPT, payload],
        capture_output=True,
        text=True,
        timeout=600,
    )

    if proc.returncode != 0:
        raise RuntimeError(
            f"Woosh worker failed (code {proc.returncode}):\n{proc.stderr}"
        )

    audio = torch.load(output_path, weights_only=True)
    os.unlink(output_path)
    if cleanup_video is not None:
        os.unlink(cleanup_video)
    return audio


def _text_conditioner_from_input(text_conditioning):
    if text_conditioning is None:
        return None
    if isinstance(text_conditioning, dict):
        return text_conditioning.get("conditioner")
    return None


def _component_dir(component):
    weights_path = getattr(component, "_weights_path", None)
    if weights_path and os.path.isfile(weights_path):
        return os.path.dirname(weights_path)
    return None


def _text_conditioner_dir_from_input(text_conditioning, component):
    if isinstance(text_conditioning, dict):
        path = text_conditioning.get("path")
        if path and os.path.isdir(path):
            return path
    return _component_dir(component)


def _validate_text_conditioner_mode(text_conditioning, model_type):
    if not isinstance(text_conditioning, dict):
        return

    mode = text_conditioning.get("mode")
    if not mode:
        return

    expected = "V2A" if model_type in ("vflow", "dvflow") else "T2A"
    if expected not in mode:
        raise RuntimeError(
            f"Connected text_conditioning is for {mode}, but this model needs {expected}. "
            "Use the matching Woosh Text Encode mode, or leave text_conditioning disconnected."
        )


def _release_module(module):
    if module is None:
        return
    try:
        module.to(_offload_device())
    except Exception:
        pass


class WooshTextEncode:
    """Load CLAP text conditioner for external use (e.g. CLAP scoring)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (
                    [
                        "T2A — text to audio (Flow/DFlow)",
                        "V2A — video to audio (VFlow/DVFlow)",
                    ],
                    {
                        "tooltip": "Pick the conditioner that matches your generative model type"
                    },
                ),
            }
        }

    RETURN_TYPES = (TEXT_COND,)
    RETURN_NAMES = ("text_conditioning",)
    FUNCTION = "encode"
    CATEGORY = "Woosh/Conditioning"
    DESCRIPTION = "Load CLAP text conditioner (use with CLAP Score node)"

    def __init__(self):
        self._model = None
        self._key = None

    def encode(self, mode):
        key = mode
        if self._model is not None and self._key == key:
            return ({"conditioner": self._model, "path": _component_dir(self._model), "mode": mode},)

        if self._model is not None:
            old_model = self._model
            self._model = None
            self._key = None
            _release_module(old_model)
            del old_model
            _clear_torch_memory()

        cond_name = "TextConditionerA" if "T2A" in mode else "TextConditionerV"
        path = _woosh_path(cond_name)
        model = SFXCLAPTextConditioner(LoadConfig(path=path))
        model.load_from_config()
        model = model.eval().to(_offload_device())

        self._model = model
        self._key = key
        return ({"conditioner": self._model, "path": path, "mode": mode},)


class WooshSample:
    """Unified sampler — auto-detects T2A vs V2A based on whether VIDEO is wired."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gen_model": (
                    "WOOSH_GEN_MODEL",
                    {"tooltip": "Generative model (Flow/DFlow/VFlow/DVFlow)"},
                ),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": True,
                        "default": "",
                        "placeholder": "e.g. sportscar engine revving and driving away quickly",
                        "tooltip": "Text prompt describing the sound to generate",
                    },
                ),
                "steps": (
                    "INT",
                    {
                        "default": 50,
                        "min": 1,
                        "max": 500,
                        "tooltip": "Number of sampling steps. Flow models: 30-100 (default 50). FlowMap (DFlow/DVFlow): 4",
                    },
                ),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 4.5,
                        "min": 0.0,
                        "max": 15.0,
                        "step": 0.1,
                        "tooltip": "Classifier-free guidance scale. Higher = more prompt adherence, lower = more creative. Default 4.5 for T2A, 3.0 for DVFlow",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFF,
                        "tooltip": "Random seed for noise generation. 0 = random each time",
                    },
                ),
                "latent_frames": (
                    "INT",
                    {
                        "default": 501,
                        "min": 1,
                        "max": 2000,
                        "tooltip": "Controls audio duration. 100 frames ≈ 1 second at 48kHz. T2A: 501≈5s, 1001≈10s. V2A: 801≈8s",
                    },
                ),
                "subprocess": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Run inference in an isolated subprocess. Use if the generated sound doesn't match the prompt — some ComfyUI environments modify global PyTorch state (attention backends, FP16 accumulation) that corrupts Woosh's output. Subprocess is slower (~15s model reload) but guaranteed correct.",
                    },
                ),
                "force_offload": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "After sampling, throw away model from GPU + CPU RAM. Next run reloads from disk",
                    },
                ),
            },
            "optional": {
                "text_conditioning": (
                    "WOOSH_TEXT_COND",
                    {
                        "tooltip": "External text conditioner (from Text Encode node). If not connected, uses model's internal CLAP"
                    },
                ),
                "video": (
                    "WOOSH_VIDEO",
                    {
                        "tooltip": "Video input for video-to-audio generation. If connected, auto-switches to V2A mode"
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", AUDIO)
    RETURN_NAMES = ("video_frames", "audio")
    FUNCTION = "sample"
    OUTPUT_IS_LIST = (False, False)
    CATEGORY = "Woosh/Sampling"
    DESCRIPTION = "Generate audio — T2A (text) or V2A (video+text) auto-detected. V2A outputs video frames for VideoCombine"

    def __init__(self):
        self._features_model = None

    def _release_features_model(self):
        if self._features_model is None:
            return
        old_model = self._features_model
        self._features_model = None
        _release_module(old_model)
        del old_model
        _clear_torch_memory()

    def _get_features_model(self, device):
        if self._features_model is None:
            self._features_model = SynchformerProcessor(frame_rate=24).eval().to(device)
        return self._features_model

    def sample(
        self,
        gen_model,
        prompt,
        cfg,
        seed,
        steps,
        latent_frames,
        subprocess=True,
        force_offload=False,
        text_conditioning=None,
        video=None,
    ):
        device = _device()
        offload = _offload_device()

        is_patcher = isinstance(gen_model, ModelPatcher)
        raw_model = gen_model.model if is_patcher else gen_model

        is_v2a = video is not None
        is_distilled = isinstance(raw_model, FlowMapFromPretrained)
        text_conditioner = _text_conditioner_from_input(text_conditioning)
        text_conditioner_dir = _text_conditioner_dir_from_input(
            text_conditioning, text_conditioner
        )

        if isinstance(raw_model, VideoKontext):
            model_type = "vflow"
        elif isinstance(raw_model, FlowMapFromPretrained):
            model_type = "dvflow" if is_v2a else "dflow"
        else:
            model_type = "flow"

        _validate_text_conditioner_mode(text_conditioning, model_type)

        if is_distilled and steps > 8:
            log.warning(
                f"[Woosh] Distilled model (DFlow/DVFlow) maxes out at 8 steps. Clamping {steps} → 8."
            )

        if is_v2a and latent_frames == 501:
            log.warning(
                f"[Woosh] V2A mode detected — auto-adjusting latent_frames from 501 (≈5s) to 801 (≈8s) for longer audio."
            )
            latent_frames = 801

        mode_label = "subprocess" if subprocess else "in-process"
        text_label = "external text conditioner" if text_conditioner else "model text conditioner"
        print(
            f'[Woosh] "{prompt}" | steps={steps} cfg={cfg} seed={seed} frames={latent_frames} [{mode_label}, {text_label}]'
        )

        try:
            mm.throw_exception_if_processing_interrupted()

            if subprocess:
                if text_conditioner is not None and text_conditioner_dir is None:
                    raise RuntimeError(
                        "External text_conditioning was provided, but its local checkpoint folder could not be determined for subprocess inference."
                    )

                model_dir = getattr(raw_model, "_weights_path", None)
                if model_dir is not None and os.path.isfile(model_dir):
                    model_dir = os.path.dirname(model_dir)
                if not model_dir or not os.path.isdir(model_dir):
                    raise RuntimeError(
                        "Cannot determine model directory for subprocess inference"
                    )

                if is_patcher:
                    gen_model.detach()

                audio = _subprocess_infer(
                    model_dir, prompt, seed, cfg, latent_frames, steps, model_type,
                    video=video if is_v2a else None,
                    text_conditioner_dir=text_conditioner_dir,
                )
            else:
                if is_patcher:
                    mm.load_model_gpu(gen_model)

                batch_size = 1
                noise = _seed_noise(
                    seed, (batch_size, LATENT_CHANNELS, latent_frames), device
                )

                batch = {"audio": None, "description": [prompt] * batch_size}

                if is_v2a:
                    features_model = self._get_features_model(device)
                    with torch.no_grad():
                        features = features_model(video["frames"], video["rate"])
                    batch["synch_out"] = features["synch_out"].expand(
                        batch_size, -1, -1
                    ).clone()

                original_text_conditioner = None
                if text_conditioner is not None:
                    if not hasattr(raw_model, "conditioners") or "text" not in raw_model.conditioners:
                        raise RuntimeError(
                            "External text_conditioning was provided, but this Woosh model has no text conditioner slot."
                        )
                    original_text_conditioner = raw_model.conditioners["text"]
                    raw_model.conditioners["text"] = text_conditioner.eval().to(device)

                try:
                    with torch.no_grad():
                        cond = raw_model.get_cond(
                            batch, no_dropout=True, device=device,
                        )
                finally:
                    if original_text_conditioner is not None:
                        raw_model.conditioners["text"] = original_text_conditioner
                        text_conditioner.to(offload)

                with torch.no_grad():
                    if is_distilled:
                        actual_steps = min(steps, 8)
                        renoise = [0, 0.5, 0.5, 0.3][:actual_steps]
                        if len(renoise) < actual_steps:
                            renoise = renoise + [0.3] * (actual_steps - len(renoise))
                        x_fake = sample_euler(
                            model=raw_model,
                            noise=noise,
                            cond=cond,
                            num_steps=actual_steps,
                            renoise=renoise,
                            cfg=cfg,
                        )
                    else:
                        from woosh.inference.flowmatching_sampler import (
                            flowmatching_integrate,
                        )

                        x_fake = flowmatching_integrate(
                            raw_model,
                            noise,
                            cond,
                            cfg=cfg,
                            device=device,
                            dtype=_flowmatching_solver_dtype(device),
                        )

                audio = raw_model.autoencoder.inverse(x_fake)
                audio = _normalize_audio(audio)

            if is_v2a and video is not None:
                frames = video["frames"]
                video_images = frames.float().cpu() / 255.0
            else:
                video_images = torch.zeros(1, 1, 1, 3, dtype=torch.float32)

        finally:
            if is_patcher:
                if force_offload:
                    gen_model.force_unload()
                    self._release_features_model()
                else:
                    gen_model.detach()

        return (video_images, {"waveform": audio.cpu(), "sample_rate": SAMPLE_RATE})


NODE_CLASS_MAPPINGS_SAMPLERS = {
    "WooshTextEncode": WooshTextEncode,
    "WooshSample": WooshSample,
}

NODE_DISPLAY_MAPPINGS_SAMPLERS = {
    "WooshTextEncode": "Woosh TextConditioning",
    "WooshSample": "Woosh Sampler",
}
