"""Woosh inference worker - runs in a clean subprocess to avoid ComfyUI's torch state.

Called by samplers.py via subprocess.run(). Takes a single JSON argument with all
parameters, loads the model, runs flow-matching ODE integration, and saves the
resulting audio tensor to disk.

This process never imports ComfyUI, so torch global state is pristine.
"""

import sys
import os
import re
import json
import torch


def _worker_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _flowmatching_solver_dtype(device):
    return torch.float32 if device.type == "mps" else torch.float64


def _patch_hf_offline(hf_cache):
    try:
        from transformers import AutoConfig, RobertaModel, RobertaTokenizer
    except ImportError:
        return

    hub_cache = os.path.join(hf_cache, "hub")
    cache_dir = hub_cache if os.path.isdir(hub_cache) else hf_cache

    orig_tok = RobertaTokenizer.from_pretrained.__func__
    orig_cfg = AutoConfig.from_pretrained.__func__
    orig_model = RobertaModel.from_pretrained.__func__

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

    def tok(cls, *args, **kwargs):
        try:
            tokenizer = orig_tok(
                cls, *args, **_kwargs(kwargs, local_files_only=True)
            )
            if _tokenizer_ok(tokenizer):
                return tokenizer
        except Exception:
            pass

        tokenizer = orig_tok(cls, *args, **_kwargs(kwargs, local_files_only=False))
        if not _tokenizer_ok(tokenizer):
            tokenizer = orig_tok(
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

    def cfg(cls, *args, **kwargs):
        try:
            config = orig_cfg(cls, *args, **_kwargs(kwargs, local_files_only=True))
            if _config_ok(config):
                return config
        except Exception:
            pass

        config = orig_cfg(cls, *args, **_kwargs(kwargs, local_files_only=False))
        if not _config_ok(config):
            config = orig_cfg(
                cls,
                *args,
                **_kwargs(kwargs, local_files_only=False, force_download=True),
            )
        return config

    def model(cls, *args, **kwargs):
        try:
            return orig_model(cls, *args, **_kwargs(kwargs, local_files_only=True))
        except Exception:
            return orig_model(cls, *args, **_kwargs(kwargs, local_files_only=False))

    RobertaTokenizer.from_pretrained = classmethod(tok)
    AutoConfig.from_pretrained = classmethod(cfg)
    RobertaModel.from_pretrained = classmethod(model)


def main():
    args = json.loads(sys.argv[1])
    model_dir = args["model_dir"]
    prompt = args["prompt"]
    seed = args["seed"]
    cfg = args["cfg"]
    latent_frames = args["latent_frames"]
    steps = args["steps"]
    is_distilled = args["is_distilled"]
    model_type = args["model_type"]
    output_path = args["output_path"]
    woosh_pkg_path = args["woosh_pkg_path"]
    hf_cache = args["hf_cache"]
    woosh_folder = args["woosh_folder"]
    woosh_folders = args.get("woosh_folders") or [woosh_folder]
    mmaudio_folders = args.get("mmaudio_folders") or []
    models_dir = args.get("models_dir")
    video_path = args.get("video_path")
    video_fps = args.get("video_fps")
    text_conditioner_dir = args.get("text_conditioner_dir")

    sys.path.insert(0, woosh_pkg_path)
    os.environ["HF_HOME"] = hf_cache
    hf_hub_cache = os.path.join(hf_cache, "hub")
    os.environ["TRANSFORMERS_CACHE"] = hf_hub_cache
    os.environ["HF_HUB_CACHE"] = hf_hub_cache
    if models_dir:
        os.environ["WOOSH_COMFYUI_MODELS_DIR"] = models_dir
    if mmaudio_folders:
        os.environ["WOOSH_MMAUDIO_DIRS"] = os.pathsep.join(mmaudio_folders)
    _patch_hf_offline(hf_cache)

    import logging

    logging.getLogger("woosh").setLevel(logging.WARNING)

    config_file = os.path.join(model_dir, "config.yaml")
    with open(config_file, "r", encoding="utf-8") as f:
        original_config = f.read()

    def _dedupe_paths(paths):
        seen = set()
        result = []
        for path in paths:
            if not path:
                continue
            full = os.path.abspath(os.path.expanduser(path))
            key = os.path.normcase(full)
            if key not in seen:
                seen.add(key)
                result.append(full)
        return result

    def _model_root_for_path(path):
        path = os.path.abspath(path)
        for root in sorted(_dedupe_paths(woosh_folders), key=len, reverse=True):
            try:
                if os.path.commonpath([path, root]) == root:
                    if path == root:
                        return os.path.dirname(root)
                    return root
            except ValueError:
                continue

        return os.path.dirname(path)

    model_root = _model_root_for_path(model_dir)

    def _resolve_woosh_path(name):
        name = os.path.normpath(str(name).replace("\\", os.sep).replace("/", os.sep))
        roots = _dedupe_paths([model_root, *woosh_folders])
        for root in roots:
            candidate = os.path.join(root, name)
            if os.path.isdir(candidate):
                return candidate
        return os.path.join(model_root, name)

    def _replace(m):
        name = m.group("name") or m.group("name2")
        path = _resolve_woosh_path(name).replace("\\", "/")
        return f"path: {path}"

    patched = re.sub(
        r"path:\s*checkpoints/(?P<name>\S+)|path:\s*\S*/models/woosh/(?P<name2>\S+)",
        _replace,
        original_config,
    )

    def _override_text_conditioner_path(content, conditioner_dir):
        if not conditioner_dir:
            return content

        conditioner_dir = os.path.abspath(conditioner_dir).replace("\\", "/")
        lines = content.splitlines()
        stack = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            indent = len(line) - len(line.lstrip(" "))
            while stack and indent <= stack[-1][0]:
                stack.pop()

            if stripped.endswith(":"):
                stack.append((indent, stripped[:-1]))
                continue

            if stripped.startswith("path:"):
                keys = [key for _, key in stack]
                if "conditioners" in keys and keys[-1:] == ["text"]:
                    lines[i] = f"{line[:indent]}path: {conditioner_dir}"

        return "\n".join(lines) + ("\n" if content.endswith("\n") else "")

    patched = _override_text_conditioner_path(patched, text_conditioner_dir)

    try:
        with open(config_file, "w", encoding="utf-8") as f:
            f.write(patched)

        from woosh.components.base import LoadConfig
        from woosh.inference.flowmatching_sampler import flowmatching_integrate
        from woosh.inference.flowmap_sampler import sample_euler

        if model_type == "vflow":
            from woosh.model.video_kontext import VideoKontext

            model = VideoKontext(LoadConfig(path=model_dir))
        elif model_type in ("dflow", "dvflow"):
            from woosh.model.flowmap_from_pretrained import FlowMapFromPretrained

            model = FlowMapFromPretrained(LoadConfig(path=model_dir))
        else:
            from woosh.model.ldm import LatentDiffusionModel

            model = LatentDiffusionModel(LoadConfig(path=model_dir))

        model = model.eval().to(_worker_device())

        device = next(model.parameters()).device

        torch.manual_seed(seed)
        noise = torch.randn(1, 128, latent_frames, device=device, dtype=torch.float32)

        batch = {"audio": None, "description": [prompt]}

        if video_path is not None and video_fps is not None:
            from woosh.utils.video import SynchformerProcessor

            frames = torch.load(video_path, weights_only=True)
            features_model = SynchformerProcessor(frame_rate=24).eval().to(device)
            with torch.no_grad():
                features = features_model(frames, video_fps)
            batch["synch_out"] = features["synch_out"].clone()

        with torch.no_grad():
            cond = model.get_cond(
                batch,
                no_dropout=True,
                device=device,
            )

        with torch.no_grad():
            if is_distilled:
                actual_steps = min(steps, 8)
                renoise = [0, 0.5, 0.5, 0.3][:actual_steps]
                if len(renoise) < actual_steps:
                    renoise = renoise + [0.3] * (actual_steps - len(renoise))
                x_fake = sample_euler(
                    model=model,
                    noise=noise,
                    cond=cond,
                    num_steps=actual_steps,
                    renoise=renoise,
                    cfg=cfg,
                )
            else:
                x_fake = flowmatching_integrate(
                    model,
                    noise,
                    cond,
                    cfg=cfg,
                    device=device,
                    dtype=_flowmatching_solver_dtype(device),
                )

        audio = model.autoencoder.inverse(x_fake)
        peak = audio.abs().amax(dim=-1, keepdim=True)
        peak = torch.clamp(peak, min=1.0)
        audio = audio / peak

        torch.save(audio.cpu(), output_path)
    finally:
        with open(config_file, "w", encoding="utf-8") as f:
            f.write(original_config)


if __name__ == "__main__":
    main()
