"""Woosh samplers — consolidated text encode + single unified sampler."""

import os
import logging
import torch
import folder_paths
import comfy.model_management as mm
from comfy.model_patcher import ModelPatcher

from woosh.inference.flowmatching_sampler import flowmatching_integrate
from woosh.inference.flowmap_sampler import sample_euler
from woosh.utils.video import SynchformerProcessor
from woosh.components.base import LoadConfig
from woosh.components.clap_conditioners import SFXCLAPTextConditioner
from woosh.model.flowmap_from_pretrained import FlowMapFromPretrained

from ..types import GEN_MODEL, TEXT_COND, VIDEO, AUDIO

log = logging.getLogger(__name__)
SAMPLE_RATE = 48000
LATENT_CHANNELS = 128
COMPRESSION_FACTOR = 480  # AE time_downscaling — 100 latent frames ≈ 1 second at 48kHz

WOOSH_FOLDER = os.path.join(folder_paths.models_dir, "woosh")


def _device():
    return mm.get_torch_device()


def _offload_device():
    return mm.intermediate_device()


def _dtype():
    return torch.float32


def _seed_noise(seed, shape, device):
    torch.manual_seed(seed)
    return torch.randn(shape, device=device)


def _normalize_audio(audio):
    peak = audio.abs().amax(dim=-1, keepdim=True)
    peak = torch.clamp(peak, min=1.0)
    return audio / peak


def _woosh_path(name: str) -> str:
    return os.path.join(WOOSH_FOLDER, name)


class WooshTextEncode:
    """Load CLAP text conditioner for external use (e.g. CLAP scoring)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["T2A — text to audio (Flow/DFlow)", "V2A — video to audio (VFlow/DVFlow)"], {"tooltip": "Pick the conditioner that matches your generative model type"}),
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
            return ({"conditioner": self._model},)

        cond_name = "TextConditionerA" if "T2A" in mode else "TextConditionerV"
        path = _woosh_path(cond_name)
        model = SFXCLAPTextConditioner(LoadConfig(path=path))
        model.load_from_config()  # loads roberta weights from TextConditionerA/weights.safetensors
        model = model.eval().to(_offload_device())

        self._model = model
        self._key = key
        return ({"conditioner": self._model},)


class WooshSample:
    """Unified sampler — auto-detects T2A vs V2A based on whether VIDEO is wired."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gen_model": ("WOOSH_GEN_MODEL", {"tooltip": "Generative model (Flow/DFlow/VFlow/DVFlow)"}),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": "", "placeholder": "e.g. sportscar engine revving and driving away quickly", "tooltip": "Text prompt describing the sound to generate"}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 500, "tooltip": "Number of sampling steps. Flow models: 30-100 (default 50). FlowMap (DFlow/DVFlow): 4"}),
                "cfg": ("FLOAT", {"default": 4.5, "min": 0.0, "max": 15.0, "step": 0.1, "tooltip": "Classifier-free guidance scale. Higher = more prompt adherence, lower = more creative. Default 4.5 for T2A, 3.0 for DVFlow"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF, "tooltip": "Random seed for noise generation. 0 = random each time"}),
                "latent_frames": ("INT", {"default": 501, "min": 1, "max": 2000, "tooltip": "Controls audio duration. 100 frames ≈ 1 second at 48kHz. T2A: 501≈5s, 1001≈10s. V2A: 801≈8s"}),
                "force_offload": ("BOOLEAN", {"default": False, "tooltip": "After sampling, throw away model from GPU + CPU RAM. Next run reloads from disk"}),
            },
            "optional": {
                "text_conditioning": ("WOOSH_TEXT_COND", {"tooltip": "External text conditioner (from Text Encode node). If not connected, uses model's internal CLAP"}),
                "video": ("WOOSH_VIDEO", {"tooltip": "Video input for video-to-audio generation. If connected, auto-switches to V2A mode"}),
            }
        }

    RETURN_TYPES = ("IMAGE", AUDIO)
    RETURN_NAMES = ("video_frames", "audio")
    FUNCTION = "sample"
    OUTPUT_IS_LIST = (False, False)
    CATEGORY = "Woosh/Sampling"
    DESCRIPTION = "Generate audio — T2A (text) or V2A (video+text) auto-detected. V2A outputs video frames for VideoCombine"

    def __init__(self):
        self._features_model = None

    def _get_features_model(self, device):
        if self._features_model is None:
            self._features_model = SynchformerProcessor(frame_rate=24).eval().to(device)
        return self._features_model

    def sample(self, gen_model, prompt, cfg, seed, steps, latent_frames, force_offload=False,
               text_conditioning=None, video=None):
        device = _device()
        offload = _offload_device()

        is_patcher = isinstance(gen_model, ModelPatcher)

        if is_patcher:
            mm.load_model_gpu(gen_model)

        raw_model = gen_model.model if is_patcher else gen_model

        try:
            mm.throw_exception_if_processing_interrupted()

            is_v2a = video is not None
            is_distilled = isinstance(raw_model, FlowMapFromPretrained)

            if is_distilled and steps > 8:
                log.warning(f"[Woosh] Distilled model (DFlow/DVFlow) maxes out at 8 steps. Clamping {steps} → 8.")

            if is_v2a and latent_frames == 501:
                log.warning(f"[Woosh] V2A mode detected — auto-adjusting latent_frames from 501 (≈5s) to 801 (≈8s) for longer audio.")
                latent_frames = 801

            batch_size = 1
            noise = _seed_noise(seed, (batch_size, LATENT_CHANNELS, latent_frames), device)

            ext_cond = None
            if text_conditioning is not None:
                ext_cond = text_conditioning["conditioner"]
                if ext_cond is not None:
                    ext_cond = ext_cond.to(device)

            cond_dict = {"audio": None, "description": [prompt] * batch_size}
            if is_v2a and video is not None:
                video_frames = video["frames"].to(device)
                video_rate = video["rate"]
                features_model = self._get_features_model(device)
                features = features_model(video_frames, video_rate)
                cond_dict["synch_out"] = features["synch_out"].expand(batch_size, -1, -1)

            if ext_cond is not None:
                text_out = ext_cond(
                    {"description": [prompt] * batch_size},
                    no_dropout=True, device=device,
                )
                cond = raw_model.get_cond(cond_dict, no_dropout=True, no_cond=False, device=device)
                cond["cross_attn_cond"] = text_out["text_cond"]
                cond["cross_attn_cond_mask"] = text_out["text_mask"]
            else:
                cond = raw_model.get_cond(cond_dict, no_dropout=True, device=device)

            with torch.inference_mode():
                mm.throw_exception_if_processing_interrupted()

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
                    x_fake = flowmatching_integrate(
                        raw_model,
                        noise=noise,
                        cond=cond,
                        cfg=cfg,
                        atol=1e-3,
                        rtol=1e-3,
                        return_steps=False,
                        device=device,
                        dtype=_dtype(),
                    )

            audio = raw_model.autoencoder.inverse(x_fake)
            audio = _normalize_audio(audio)

            # Prepare video frames output for V2A (ComfyUI IMAGE: [B, H, W, C] float32 [0,1])
            if is_v2a and video is not None:
                frames = video["frames"]  # (T, H, W, C) uint8
                video_images = frames.float().cpu() / 255.0  # (T, H, W, C) float32 [0,1] on CPU
            else:
                video_images = torch.zeros(1, 1, 1, 3, dtype=torch.float32)

        finally:
            if is_patcher:
                if force_offload:
                    gen_model.force_unload()
                else:
                    gen_model.detach()
            if ext_cond is not None and text_conditioning is not None:
                text_conditioning["conditioner"].to(offload)

        return (video_images, {"waveform": audio.cpu(), "sample_rate": SAMPLE_RATE})


NODE_CLASS_MAPPINGS_SAMPLERS = {
    "WooshTextEncode": WooshTextEncode,
    "WooshSample": WooshSample,
}

NODE_DISPLAY_MAPPINGS_SAMPLERS = {
    "WooshTextEncode": "Woosh TextConditioning",
    "WooshSample": "Woosh Sampler",
}
