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
    video_path = args.get("video_path")
    video_fps = args.get("video_fps")

    sys.path.insert(0, woosh_pkg_path)
    os.environ["HF_HOME"] = hf_cache
    os.environ["TRANSFORMERS_CACHE"] = hf_cache
    os.environ["HF_HUB_CACHE"] = os.path.join(hf_cache, "hub")

    import logging

    logging.getLogger("woosh").setLevel(logging.WARNING)

    config_file = os.path.join(model_dir, "config.yaml")
    with open(config_file, "r", encoding="utf-8") as f:
        original_config = f.read()

    woosh_url = woosh_folder.replace("\\", "/")

    def _replace(m):
        name = m.group("name") or m.group("name2")
        return f"path: {woosh_url}/{name}"

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

        model = model.eval().cuda()

        device = next(model.parameters()).device

        torch.manual_seed(seed)
        noise = torch.randn(1, 128, latent_frames, device=device)

        batch = {"audio": None, "description": [prompt]}

        if video_path is not None and video_fps is not None:
            from woosh.utils.video import SynchformerProcessor

            frames = torch.load(video_path, weights_only=True)
            features_model = SynchformerProcessor(frame_rate=24).eval().to(device)
            with torch.no_grad():
                features = features_model(frames, video_fps)
            batch["synch_out"] = features["synch_out"]

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
