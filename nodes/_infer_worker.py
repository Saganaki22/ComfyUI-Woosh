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


def _patch_hf_offline(hf_cache):
    try:
        from transformers import AutoConfig, RobertaModel, RobertaTokenizer
    except ImportError:
        return

    orig_tok = RobertaTokenizer.from_pretrained.__func__
    orig_cfg = AutoConfig.from_pretrained.__func__
    orig_model = RobertaModel.from_pretrained.__func__

    def tok(cls, *args, **kwargs):
        kwargs.setdefault("cache_dir", hf_cache)
        try:
            return orig_tok(cls, *args, local_files_only=True, **kwargs)
        except Exception:
            return orig_tok(cls, *args, **kwargs)

    def cfg(cls, *args, **kwargs):
        kwargs.setdefault("cache_dir", hf_cache)
        try:
            return orig_cfg(cls, *args, local_files_only=True, **kwargs)
        except Exception:
            return orig_cfg(cls, *args, **kwargs)

    def model(cls, *args, **kwargs):
        kwargs.setdefault("cache_dir", hf_cache)
        try:
            return orig_model(cls, *args, local_files_only=True, **kwargs)
        except Exception:
            return orig_model(cls, *args, **kwargs)

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

    sys.path.insert(0, woosh_pkg_path)
    os.environ["HF_HOME"] = hf_cache
    os.environ["TRANSFORMERS_CACHE"] = hf_cache
    os.environ["HF_HUB_CACHE"] = os.path.join(hf_cache, "hub")
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
                    model, noise, cond, cfg=cfg, device=device
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
